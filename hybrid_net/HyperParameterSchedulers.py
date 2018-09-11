import numpy as np

def linear_rampdown(current, rampdown_length):
    """Linear rampdown"""
    if current >= rampdown_length:
        return 1.0 - current / rampdown_length
    else:
        return 1.0

def adjust_lambda_r(current, rampup_length, rampdown_length, total_epochs):
    if current <= rampup_length:
        phase = 1 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    elif current >= rampdown_length:
        phase = (current - rampdown_length) / (total_epochs - rampdown_length)
        return float(np.exp(-5.0 * phase * phase))
    else:
        return 1.0

def exponential_decrease(current, rampdown_length, total_epochs, scale = 1.0):
    if current <= rampdown_length:
        return 1.0
    else:
        phase = scale * (current - rampdown_length) / (total_epochs - rampdown_length)
        return float(np.exp(-5.0 * phase * phase))

def exponential_increase(current, rampup_length):
    if current >= rampup_length:
        return 1.0
    else:
        phase = 1 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def adjust_learning_rate(optimizer, initial_lr, epoch, total_epochs):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = initial_lr * linear_rampdown(epoch, total_epochs - total_epochs/3)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_beta_1(optimizer, initial_beta1, epoch, total_epochs):
    beta1 = initial_beta1 * exponential_decrease(epoch, 0.8 * total_epochs, total_epochs, scale = 0.5)
    
    for param_group in optimizer.param_groups:
        _ , beta2 = param_group['betas']
        param_group['betas'] = (beta1, beta2)