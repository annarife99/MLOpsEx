from __init__ import _PATH_DATA
import data
from MLOpsEx.data import mnist

def test_data():
    
   dataset_train,dataset_test = mnist()
   print(dataset_train['images'].shape)

   #assert len(dataset_train) == N_train 
   #for training and N_test for test
   #assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
   #assert that all labels are represented

test_data()