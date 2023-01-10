import argparse
import sys

import click
import torch
from torch import nn

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-1, help='learning rate to use for training')


def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel(784, 10, [512, 256, 128])
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # TODO: Implement training loop here
    train_set, _ = mnist()
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=64)
    
    steps = 0
    epochs=5

    for e in range(epochs):
        running_loss = 0
        model.train()
        # Model in training mode, dropout is on
        for images,labels in dataloader:
            steps += 1
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

    torch.save(model.state_dict(), 'trained_model.pt')
   


@click.command()
@click.argument("model_checkpoint")

def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    
    state_dict = torch.load(model_checkpoint)
    model=MyAwesomeModel(784, 10, [512, 256, 128])
    model.load_state_dict(state_dict)
    
    criterion = nn.NLLLoss()
    _, test_set = mnist()
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)

    accuracy = 0
    test_loss = 0

    for images, labels in dataloader:
        images = images.resize_(images.size()[0], 784)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    print(accuracy)
    print('Test Loss:',test_loss)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    