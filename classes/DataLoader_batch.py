import torch

class DataLoader_batch:
    """DataLoader iterator class to provide model with batches of data
    """

    def __init__(self, tab: list, Y: list, batch_size: int) -> None:
        self.tab = tab
        self.Y = Y
        self.batch_size = batch_size
        self.counter = 0
        self.n = len(tab)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.n:
            self.counter += self.batch_size
            batch_X = self.tab[self.counter - self.batch_size:self.counter]
            batch_Y = self.Y[self.counter - self.batch_size:self.counter]
            return torch.Tensor(batch_X), torch.Tensor(batch_Y)

        raise StopIteration

    def __len__(self):
        return self.n // self.batch_size