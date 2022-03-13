# General imports
from torch.nn.utils import prune
import numpy as np

def pruning_mutation(type_of_pruning, layers_to_prune, pruning_amount):
    """
    The function that gets called from outside classes and handles which type of pruning 
    is going to be executed by calling the other functions.
    """
    if type_of_pruning == "ru":
        unstrcutured(prune.RandomUnstructured, layers_to_prune, pruning_amount)
    elif type_of_pruning == "rs":
        structured(layers_to_prune, pruning_amount)
    elif type_of_pruning == "l1u":
        unstrcutured(prune.L1Unstructured, layers_to_prune, pruning_amount)

def pruning_types():
    """
    Returns a list of all possible implemented pruning methods. Can be easily extended 
    by defining a function and adding a new element to the list that is being returned.
    """
    return ["ru", "rs", "l1u"]

def unstrcutured(method, layers_to_prune, pruning_amount):
    """
    Prunes *pruning_amount* (percentage; decimal number) of the weights in 
    *layers_to_prune* using unstructured pruning strategy *method*.
    """
    # Set random weights to "pruned"
    prune.global_unstructured(
                [(layer, "weight") for layer in layers_to_prune],
                pruning_method=method,
                amount=pruning_amount,
            )
    # Actually remove the "pruned" weights
    for m in layers_to_prune:
        if prune.is_pruned(m):
            prune.remove(m, "weight")

def structured(layers_to_prune, pruning_amount):
    """
    Prunes *pruning_amount* (number; an integer) of the channels in *layers_to_prune* 
    using structured pruning.
    """
    layer_nr = np.random.randint(0, len(layers_to_prune))
    m = prune.random_structured(layers_to_prune[layer_nr], 'weight', amount=pruning_amount, dim=0)
    prune.remove(m, "weight")
    layers_to_prune[layer_nr] = m