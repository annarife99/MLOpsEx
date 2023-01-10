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
@click.option("--lr", default=1e-1, help='learning rate to use for training')


def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel(784, 10, [512, 256, 128])
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # TODO: Implement training loop here
    
    # Access data from processed folder
    train_data = torch.load("data/processed/train.pth",)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)

    steps = 0
    epochs=20

    for e in range(epochs):
        running_loss = 0
        model.train()
        # Model in training mode, dropout is on
        for images,labels in dataloader:
            steps += 1
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            images=images.to(torch.float32)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

    #save figures.png in reports/figures/

    torch.save(model.state_dict(), 'models/trained_model.pt')
   


cli.add_command(train)


if __name__ == "__main__":
    cli()







    
    
    
    