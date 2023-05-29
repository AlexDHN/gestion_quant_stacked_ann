import torch
import torch.nn as nn
from typing import Callable

class Ann_extension(nn.Module):
  
  def __init__(self,input_size,output_size):
    super(Ann_extension,self).__init__()
    self.output = nn.LSTM(input_size,output_size,3)
    self.weights_are_initialized = False  # The weights are not yet initialized
  
  def forward(self,x):
    assert self.weights_are_initialized, 'The weights are not initialized'
    return self.output(x)[0].reshape(-1)

  def init_weights(self, f: Callable):
    """ Initialize the weights according to a distribution f defines

    Args:
        f (Callable): Function that associates to each weight a certain distribution
    """
    f(self)
    self.weights_are_initialized = True  # The weights are now initialized

  def save(self, path: str):
    """ Save the Ann of the model
    """
    torch.save(self, path)  #torch.save(self.state_dict(), path)
    
  @classmethod
  def load(cls, path: str):
      """ Load the pretrain Ann of the model
      """
      model = torch.load(path)  # .load_state_dict(torch.load(path))
      model.eval() 
      return model 