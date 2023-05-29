from classes.Simulation import Simulation
from utils.init_functions import random_normal
import numpy as np
from copy import deepcopy

def grid_search(learning_rate,phi:list, pivot:int,**kwargs):
    min = np.inf
    for lr in learning_rate:
        for lmd in phi:
            kwargs["learning_rate"] = lr
            kwargs["l2_lambda"] = lmd
            sim = Simulation(**kwargs)
            sim.Ann.init_weights(random_normal)
            sim.make_dataloaders(pivot_index=pivot)
            sim.train(verbose=1)
            rmse = np.sqrt(sim.test_loss_history[-1])
            if rmse < min:
                min = rmse
                best_lr = lr
                best_phi = lmd
                best_sim = deepcopy(sim)
    best_sim.Ann.save("models/torch/{}".format(kwargs["period"]))
    return best_lr, best_phi