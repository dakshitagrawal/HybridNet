import os
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from hybrid_net import DataTransformer
from hybrid_net import DataLoaderMaker
from hybrid_net import ModelArchitecture
from hybrid_net import MetricMeters
from hybrid_net import HyperParameterSchedulers
from hybrid_net import Losses

from PIL import Image


images = "./data-local/images/cifar10/by-image/"
labels = "./data-local/labels/cifar10/1000_balanced_labels/00.txt"
train_subdir = "train"
eval_subdir = "val"
checkpoint_path = "./checkpoints/"
device = "cuda" if torch.cuda.is_available() else 'cpu'

workers = 4
NO_LABEL = -1
global_step = 0
ema_decay = 0.999
print_freq = 2
best_prec1 = 0
initial_lr = 0.003
initial_beta1 = 0.9
start_epoch = 0
arch = "hybridNet1"

# batch_size is 100 in paper
batch_size = 10

# batch_size is 20 in paper
labeled_batch_size = 1

total_epochs = 3

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    print('--- checkpoint saved to %s ---'.format(checkpoint_path))
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print('--- checkpoint copied to %s ---'.format(best_path))

train_transformation, eval_transformation = DataTransformer.transformer()

train_loader, eval_loader = DataLoaderMaker.create_data_loaders(train_transformation, 
                                                eval_transformation, 
                                                images, 
                                                train_subdir, 
                                                eval_subdir,
                                                labels,
                                                batch_size,
                                                labeled_batch_size,
                                                workers)

model = ModelArchitecture.create_model()
ema_model = ModelArchitecture.create_model(ema=True)

def train(train_loader, model, ema_model, optimizer, epoch, ema_decay, total_epochs, print_freq = 2):
    global global_step

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).to(device)
    reconstruction_criterion = Losses.symmetric_mse_loss   
    stability_criterion = Losses.softmax_mse_loss

    meters = MetricMeters.AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i, ((input, ema_input), target) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)
        
        HyperParameterSchedulers.adjust_learning_rate(optimizer, initial_lr, epoch, total_epochs)
        HyperParameterSchedulers.adjust_beta_1(optimizer, initial_beta1, epoch, total_epochs)
        lambda_c = HyperParameterSchedulers.exponential_increase(i, 800)
        lambda_r = 100 * HyperParameterSchedulers.adjust_lambda_r(epoch, 0.25 * total_epochs, 0.8 * total_epochs, total_epochs)
        lambda_s = HyperParameterSchedulers.exponential_decrease(epoch, 0.95 * total_epochs, total_epochs)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        meters.update('beta1', optimizer.param_groups[0]['betas'][0])
        meters.update('lambda_c', lambda_c)
        meters.update('lambda_r', lambda_r)
        meters.update('lambda_s', lambda_s)

        input_var = torch.autograd.Variable(input)
        ema_input_var = torch.autograd.Variable(ema_input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True))

        input_var = input_var.to(device)
        ema_input_var = ema_input_var.to(device)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().float()
        
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        ema_y, ema_x_c, ema_x_u = ema_model(ema_input_var)
        model_y, model_x_c, model_x_u = model(input_var)

        ema_logit = Variable(ema_y.detach().data, requires_grad=False)
        cons_logit = model_y
        class_logit = model_y
        
        # classification loss
        class_loss = class_criterion(class_logit, target_var) / labeled_minibatch_size
        meters.update('class_loss', class_loss.data[0])
        
        ema_class_loss = class_criterion(ema_logit, target_var) / labeled_minibatch_size
        meters.update('ema_class_loss', ema_class_loss.data[0])

        # reconstruction loss
        reconstruction_loss = reconstruction_criterion(model_x_c + model_x_u, input_var)/minibatch_size
        meters.update('reconstruction_loss', reconstruction_loss)
        
        # TODO add reconstruction loss of in-between layers
        #
        #
        #
        
        # balanced reconstruction loss
        if torch.sum((model_x_c - input_var)**2) <= torch.sum((model_x_u - input_var)**2):
            reconstruction_loss = reconstruction_criterion(model_x_u + model_x_c.detach(), input_var)/minibatch_size
        else:
            reconstruction_loss = reconstruction_criterion(model_x_u.detach() + model_x_c, input_var)/minibatch_size
      
        
        # stability loss
        stability_loss = stability_criterion(cons_logit, ema_logit) / minibatch_size
        meters.update('stability_loss', stability_loss.data[0])


        loss = lambda_c * class_loss + lambda_r * reconstruction_loss + lambda_s * stability_loss 
        
        
        assert not (np.isnan(loss.data[0]) or loss.data[0] > 1e5), 'Loss explosion: {}'.format(loss.data[0])
        
        meters.update('loss', loss.data[0])

        prec1, prec5 = MetricMeters.accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        ema_prec1, ema_prec5 = MetricMeters.accuracy(ema_logit.data, target_var.data, topk=(1, 5))
        meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_top5', ema_prec5[0], labeled_minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        
        update_ema_variables(model, ema_model, ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))

def validate(eval_loader, model, global_step, epoch, print_freq = 2):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).to(device)
    meters = MetricMeters.AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        input_var = input_var.to(device)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().float()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        model_y, model_x_c, model_x_u = model(input_var)
        softmax_model_y = F.softmax(model_y, dim=1)
        class_loss = class_criterion(model_y, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = MetricMeters.accuracy(model_y.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                'Test: [{0}/{1}]\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    i, len(eval_loader), meters=meters))

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))

    return meters['top1'].avg


optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)

for epoch in range(start_epoch, total_epochs):
    start_time = time.time()
    train(train_loader, model, ema_model, optimizer, epoch, ema_decay, total_epochs, print_freq)
    time_elapsed = time.time() - start_time
    print('epoch training complete in {: .0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    start_time = time.time()
    print("Evaluating the primary model:")
    prec1 = validate(eval_loader, model, global_step, epoch, print_freq = 2)
    print("Evaluating the EMA model:")
    ema_prec1 = validate(eval_loader, model, global_step, epoch + 1, print_freq = 2)
    time_elapsed = time.time() - start_time
    print('epoch validation complete in {: .0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    is_best = ema_prec1 > best_prec1
    best_prec1 = max(ema_prec1, best_prec1)
    
    
    save_checkpoint({
        'epoch': epoch + 1,
        'global_step': global_step,
        'arch': arch,
        'state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_path, epoch + 1)

