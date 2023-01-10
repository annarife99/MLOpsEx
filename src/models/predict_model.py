import argparse
import sys
import numpy as np

import torch
from torch import nn
import click
from model import MyAwesomeModel


@click.group()
def cli():
    pass



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

    test_data = torch.load("data/processed/test.pth")
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True)


    accuracy = 0
    test_loss = 0

    for images, labels in dataloader:
        images = images.resize_(images.size()[0], 784)
        images=images.to(torch.float32)
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


cli.add_command(evaluate)



if __name__ == "__main__":
    cli()
