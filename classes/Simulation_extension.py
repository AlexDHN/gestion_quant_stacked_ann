import torch
import pandas as pd
import numpy as np

from time import time
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt

from classes.Ann_extension import Ann_extension
from classes.DataLoader_batch import DataLoader_batch


class Simulation_extension:
    def __init__(self, **kwargs) -> None:
        self.learning_rate = kwargs["learning_rate"]
        self.batch_size = kwargs["batch_size"]
        self.num_epochs = kwargs["num_epochs"]
        self.window_size = kwargs["window_size"]
        self.weight_decay = kwargs["weight_decay"]
        self.tab = kwargs["tab"]
        self.Y = kwargs["y"]
        
        if "path" in kwargs:
            self.Ann_extension = Ann_extension.load(kwargs["path"])  # It is assumed that the characteristics of the loaded Ann_extension correspond to the input data
        else:
            self.Ann_extension = Ann_extension(
                input_size = self.window_size+3,  
                output_size = 1, 
            )
        
        self.optimizer = Adam(
            params=self.Ann_extension.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay, 
        )
        self.criterion = torch.nn.MSELoss()
    
    def make_dataloaders(self, pivot_index):
        """_summary_

        Args:
            pivot_index (int): Pivot to identify the training and test data set
        """
        
        self.train_dataloader = lambda : DataLoader_batch(
            tab=self.tab[:pivot_index], 
            Y=self.Y[:pivot_index],
            batch_size=self.batch_size,
        )
        
        self.test_dataloader = lambda : DataLoader_batch(
            tab=self.tab[pivot_index:],
            Y=self.Y[pivot_index:],
            batch_size=self.batch_size,
        )
        
        self.dataloaders = {
            "train": self.train_dataloader, 
            "test": self.test_dataloader, 
        }
    
    def train(self, verbose: int = 1):
        """ Main training loop
        
        Args:
            verbose (int, optional): 0: no output, 1: epochs tqdm, 2: prints losses per epoch. Defaults to 1.
        """
        self.train_loss_history = []
        self.test_loss_history = []
        main_loop = tqdm(range(self.num_epochs)) if verbose == 1 else range(self.num_epochs)
        for epoch in main_loop:
            start_time = time()
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.Ann_extension.train()
                else:
                    self.Ann_extension.eval()
                loss_sum = 0 # Total loss
                loss_num = 0 # To compute average loss
                
                if phase == 'train':
                    iterator = tqdm(self.dataloaders[phase]()) if verbose == 2 else self.dataloaders[phase]()
                else:
                    iterator = self.dataloaders[phase]()
                for batch in iterator:
                    x, y_true = batch
                    if phase == 'train' and verbose == 2:
                        iterator.set_description(f"Epoch {epoch+1}/{self.num_epochs}")
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        
                        y_pred = self.Ann_extension(x)
                        loss = self.criterion(y_pred, y_true)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        
                        loss_sum += loss.item()
                        loss_num += 1
                
                # Print loss if verbose == 2
                if verbose == 2:
                    if phase == 'train':
                        time_elapsed = time() - start_time
                        print(f'Epoch {epoch+1}/{self.num_epochs} complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
                    print("  ==> {} loss: {:.4f}".format(phase.capitalize(), loss_sum / loss_num * 100))
                    print()
                
                # Save loss history
                if phase == 'train':
                    self.train_loss_history.append(loss_sum / loss_num)
                else:
                    self.test_loss_history.append(loss_sum / loss_num)
            if verbose==1:
                main_loop.set_description(f"Train loss: {self.train_loss_history[-1]*100:.4f} | Test loss: {self.test_loss_history[-1]*100:.4f}")
            
    def plot_loss(self):
        """ Plots the loss history for the training and test sets
        """
        plt.plot(self.train_loss_history, label="Train")
        plt.plot(self.test_loss_history, label="Test")
        plt.legend()
        plt.show()