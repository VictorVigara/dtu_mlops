import argparse
import sys

import torch
import torch.nn as nn
from  torch.utils.data import DataLoader 
from torch import  optim

import click

import data
from model import Net

from sklearn import metrics

import matplotlib.pyplot as plt

import os


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--ep", default=30, help='epochs for training')
@click.option("--bs", default=32, help='batch size')
def train(lr, bs, ep):
    print("Starting training")
    print(f"Learning rate: {lr}  Batch size: {bs}  Epochs: {ep}")
    curr_dir = os.getcwd()

    # Path to data
    data_path = "C:/Users/victo/OneDrive/Escritorio/DTU/Machine_Learning_Operations/dtu_mlops/data/corruptmnist"

    # Define model
    model = Net(784, [256, 132, 100], 10)
    # Load train data
    train_set = data.MyDataset(data_path, 'train')
    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)

    # Load test data
    test_set = data.MyDataset(data_path, 'test')
    testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 20
    steps = 0
    best_test_loss = 0

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for e in range(ep):

        # TRAINING
        train_curr_loss = 0
        train_preds, train_targs = [], []

        model.train()
        for images, label in trainloader:
            # Reshape image to enter FFNN
            images = images.view(images.shape[0], -1)
            # Set gradients to zero
            optimizer.zero_grad()
            # Predict 
            output = model(images)
            t_preds = torch.max(output,1)[1]
            # Calculate loss
            loss = criterion(output, label)
            # Calculate gradients
            loss.backward()
            # Update weights
            optimizer.step()
            # Get loss from all batch
            train_curr_loss += loss.item()

            train_targs += list(label.numpy())
            train_preds += list(t_preds.numpy())
        # Compute train batch loss and accuracy
        train_batch_loss = train_curr_loss/len(trainloader)
        train_loss.append(train_batch_loss)

        train_batch_acc = metrics.accuracy_score(train_targs, train_preds)
        train_acc.append(train_batch_acc)

        # VALIDATION
        test_curr_loss = 0
        test_preds, test_targs = [], []
        model.eval()
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            # Predict 
            output = model(images)
            te_preds = torch.max(output,1)[1]
            # Calculate loss
            loss = criterion(output, labels)

            test_curr_loss += loss.item()

            test_targs += list(labels.numpy())
            test_preds += list(te_preds.numpy())

        # Compute test batch loss and accuracy
        test_batch_loss = test_curr_loss/(len(testloader))    
        test_loss.append(test_batch_loss)

        test_batch_acc = metrics.accuracy_score(test_targs, test_preds)
        test_acc.append(test_batch_acc)
        
        print(f"Epoch {e+1} Train Loss =  {train_batch_loss}  Train Acc = {train_batch_acc} Test Loss  =  {test_batch_loss}  Test Acc  = {test_batch_acc}")
        print("--------------------------------------------------------------------------")

        # Save best model (lowest test loss)
        if test_batch_loss > best_test_loss: 
            best_test_loss = test_batch_loss
            save_dir = curr_dir+"/s1_development_environment/exercise_files/final_exercise/best_checkpoint.pth"
            torch.save(model.state_dict(), save_dir)
    '''
    plt.figure()
    print()
    plt.plot(ep, train_loss, 'b', test_loss, 'r')
    plt.legend(['Train Loss','Test Loss'])
    plt.xlabel('Updates'), plt.ylabel('Acc')

    plt.figure()
    plt.plot(ep, train_acc, 'b', test_acc, 'r')
    plt.legend(['Train accuracy', 'Test accuracy'])
    plt.xlabel('Updates'), plt.ylabel('Acc')
    '''
@click.command()
@click.argument("model_checkpoint")
@click.option("--bs", default=32, help='batch size')

def evaluate(model_checkpoint, bs):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)
    curr_dir = os.getcwd()

    # Path to data
    data_path = "C:/Users/victo/OneDrive/Escritorio/DTU/Machine_Learning_Operations/dtu_mlops/data/corruptmnist"

    # Load model
    model = Net(784, [256, 132, 100], 10)

    # Load parameters to model
    checkpoint_dir = curr_dir+"/s1_development_environment/exercise_files/final_exercise/"+model_checkpoint
    state_dict = torch.load(checkpoint_dir)
    model.load_state_dict(state_dict)
    
    # Load test data
    test_set = data.MyDataset(data_path, 'test')
    testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

    images, labels = next(iter(testloader))
    # Get the class probabilities
    ps = torch.exp(model(images))
    # Most likely classes 
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    # Get model accuracy
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')

    

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  