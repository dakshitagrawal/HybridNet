# Pytorch Implementation of HybridNet

This repository contains the implementation of the HybridNet model introduced in the paper `"HybridNet: Classification and Reconstruction Cooperation for Semi-Supervised Learning"`.

The paper can be found at this [**link**](https://arxiv.org/abs/1807.11407).

## Dependencies

The following are the dependencies required by the repository:

+ pytorch v0.4
+ numpy 
+ scipy 
+ pandas 
+ tqdm 
+ matplotlib
+ PIL

## Setup Instructions

The repository can be setup easily on your local system if you have all the dependencies satisfied.

First download the repository on your local machine by either downloading it or running the following line on `cmd prompt`.

``` Batchfile
git clone https://github.com/dakshitagrawal97/HybridNet.git
```

Due to the large size of CIFAR-10 dataset, it has not been stored in the repository.  The repository expects the images of the dataset to be in the `data-local` folder.  You may set up CIFAR-10 inside the repository by running the following command.

``` shell
./data-local/bin/prepare_cifar10.sh
```

## Training Instructions

The hyperparameters of the model have been set within the codebase according to the paper.  To run the model for training, simply run the following command.

``` cmd
python main.py
```

You may also have an interactive session with the code through the Jupyter Notebook `Main_Train.ipynb`.

## TO-DO

1. [ ] Train the model and save checkpoints.
2. [ ] Build ConvLarge Network for STL-10 dataset and train model.
3. [ ] Build ResNet Network and train model.
4. [ ] Perform ablation studies as mentioned in the paper.
