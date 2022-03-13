# General imports
import torch 
import numpy as np
import copy

class Net():
    """
    The class representing an individual neural network, made for ease of accessing a network's features.
    It has the following attributes:
    * model        - the network itself (structure, weights, and biases),
    * accuracies   - a list of the accuracies of the network on the datasets it was tested on,
    * loss         - the cross entropy loss of the network,
    * fitness      - the fitness of the network,
    * weight_vals  - a list of the shared weight values with which *accuracies* are achieved.
    """
    def __init__(self, model, accuracies=[], loss=float("inf"), fitness=0, values=[]):
        self.model = model
        self.accuracies = accuracies
        self.loss = loss
        self.fitness = fitness
        self.weight_vals = values
    
    def get_nonzero_edges(self):
        """
        Returns the number of non-zero edges in the network.
        """
        edges = 0
        for child in self.get_all_layers():
            d = copy.deepcopy(child.state_dict())
            for g in d:
                if g == 'weight':
                    edges += (torch.count_nonzero(d[g])).item()
        return edges
    
    def model_stats(self):
        """
        Returns a description of the network and its features.
        """
        accuracies = ""
        for ind in range(len(self.accuracies)):
            accuracies += "\nDataset " + str(ind+1) + ": " + str(self.accuracies[ind]) + "% achieved with weight value = " + str(self.weight_vals[ind])
        return "The model has " + str(self.get_nonzero_edges()) + " edges\nfitness: " + str(self.fitness) + "\nloss: " + str(self.loss) + accuracies

    def evaluate(self, data_loader):
        """
        Evaluates the model on dataset *data_loader* and returns its accuracy and loss.
        The attributes of the Net do not get updated, however.
        """
        # Setup for testing
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        loss = 0
        accuracy = 0
        targets = []
        all_outputs = []
        # Predicting and calculating loss and accuracy
        with torch.no_grad():
            for batch in data_loader: 
                # Predict
                data, labels = batch
                outputs = self.model(data)
                loss += criterion(outputs, labels).item()
                predicted = outputs.argmax(dim=1, keepdim=True)
                accuracy += predicted.eq(labels.view_as(predicted)).sum().item()
                # Bookkeeping
                targets.append(labels.numpy())
                all_outputs.append(predicted.numpy())

        # Calculating the average accuracy and loss over all samples and returning them
        avg_accuracy = accuracy/len(data_loader.dataset)
        avg_loss = loss/len(data_loader.dataset)
        return avg_loss, avg_accuracy*100, np.concatenate(targets).ravel(), np.concatenate(all_outputs).ravel()

    def make_copy(self):
        """
        Useful method for creating and returning a deep copy of the network.
        """
        return Net(copy.deepcopy(self.model), self.accuracies, self.loss, self.fitness, self.weight_vals)

    """
    A helper method, which returns all layers within the network, flattened (i.e. the layers within the 
    Sequential layers are added to the list of layers instead of just their parent Sequential layer).
    NB: The InitialModel we use currently does not need any "flattening" but we aim to make the code as
    general and reusable as possible.
    """
    def get_all_layers(self):
        all_children = []
        for child in self.model.children():
            children = get_children_of_module(child)
            if isinstance(children, list):
                for c in children:
                    all_children.append(c)
            else:
                all_children.append(children)
        return all_children

"""
A helper method, which unwraps a Sequential layer and returns the layers within it (and for other layers, 
just returns the layer itself).
NB: The InitialModel we use currently does not need any "unwrapping" since it does not have any Sequential
layers, but we aim to make the code as general and reusable as possible.
Source of code: https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
"""
def get_children_of_module(model: torch.nn.Module):
    children = list(model.children())
    flattened_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flattened_children.extend(get_children_of_module(child))
            except TypeError:
                flattened_children.append(get_children_of_module(child))
    return flattened_children