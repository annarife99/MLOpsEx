# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np
from torchvision.transforms import transforms
from torchvision import transforms


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_paths = [input_filepath + f"/train_{i}.npz" for i in range(5)]
    X_train = torch.from_numpy(np.concatenate([np.load(train_file)["images"] for train_file in train_paths]))
    Y_train = torch.LongTensor(np.concatenate([np.load(train_file)["labels"] for train_file in train_paths]))

    X_test = torch.from_numpy(np.load(input_filepath + "/test.npz")["images"])
    Y_test = torch.LongTensor(np.load(input_filepath + "/test.npz")["labels"])


    normalize = transforms.Normalize(0, 1)
    # normalize the image tensor
    nor_X_train = normalize(X_train)
    nor_X_test = normalize(X_train)


    train = list(zip(nor_X_train.reshape(-1, 1, 28, 28), Y_train))
    test = list(zip(nor_X_test.reshape(-1, 1, 28, 28), Y_test))
    

    torch.save(train, output_filepath + "/train.pth")
    torch.save(test, output_filepath + "/test.pth")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
