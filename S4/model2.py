import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningModule

class MyAwesomeModel(LightningModule):
    def __init__(self,input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()

        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
    
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.resize_(x.size()[0], 784)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y== y_hat.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc) 
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-1)





