# General imports
import torch
import copy
import argparse
import sys

# Custom imports
from Net import Net
from InitialModel import InitialModel
from DatasetManager import DatasetManager
from Pruner import pruning_mutation, pruning_types

class NoGA():
    """
    A class that facilitates running the ablation study of pruning the initial network, evaluating
    the pruned model and comparing it to PADAWANN. The attributes are:
    * initial_model   - the network used for creating the initial population in the GA,
    * padawann        - the PADAWANN obtained from that initial model,
    * pruning_amount  - the percentage of pruned edges of the PADAWANN,
    * dataset_manager - a data structure containing the datasets used for testing.
    """
    def __init__(self, initial_model=None, padawann=None, datasets=[]):
        # Loading the models' weights and biases
        pretrained_model = InitialModel()
        pretrained_model.load_state_dict(torch.load(initial_model))
        self.initial_model = Net(pretrained_model)

        padawann_model = InitialModel()
        padawann_model.load_state_dict(torch.load(padawann))
        self.padawann = Net(padawann_model)

        # How much to prune
        self.pruning_amount = 1.0 - self.padawann.get_nonzero_edges()/self.initial_model.get_nonzero_edges()

        # Datasets for evaluation
        self.dataset_manager = DatasetManager()
        self.dataset_manager.load_datasets(datasets, "test")

    def pruning(self):
        """
        Prunes *self.initial_model* in all available ways, evaluates the pruned models' performance 
        and compares its structure to that of the *self.padawann*.
        """
        for pruning_type in pruning_types():
            sys.stdout.write("\nUsing %s pruning:" % pruning_type)
            model_copy = self.initial_model.make_copy()
            # Make a list of all prune-able layers
            layers = []
            for layer in model_copy.get_all_layers():
                if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Conv3d):
                    layers.append(layer)
            # Make pruned model and store it
            pruning_mutation(pruning_type, layers, self.pruning_amount)
            torch.save(model_copy.model.state_dict(), "pruned_" + pruning_type + '.pth')
            # Evaluate it and compare its structure to PADAWANN
            self.evaluate(model_copy)
            sys.stdout.write("\nPNN has %d edges" % model_copy.get_nonzero_edges())
            self.edge_comparison(model_copy)

    def evaluate(self, model):
        """
        Evaluates *model* on all datasets loaded in *self.dataset_manager* and prints the results.
        """
        for dataset_nr in range(0, self.dataset_manager.get_number_of_tasks()):
            dataset = self.dataset_manager.create_loader(dataset_nr, "test")
            _, accuracy, _, _ = model.evaluate(dataset)
            sys.stdout.write("\nAccuracy on dataset #%(num)d: %(acc)f" % {"num" : dataset_nr+1, "acc" : accuracy})

    def edge_comparison(self, model):
        """
        Compares the structure (non-zero connections) of the pruned model *model* and the 
        *self.padawann* and returns the number of edges they have pruned in common.
        """
        pruned_model_edges = []
        for child in model.get_all_layers():
            d = copy.deepcopy(child.state_dict())
            if 'weight' in d:
                pruned_model_edges.append(copy.deepcopy(child.state_dict())['weight'].numpy())
                
        padawann_edges = []
        for child in self.padawann.get_all_layers():
            d = copy.deepcopy(child.state_dict())
            if 'weight' in d:
                padawann_edges.append(copy.deepcopy(child.state_dict())['weight'].numpy())
                
        same = 0
        for i in range(0, len(pruned_model_edges)):
            same += sum(x == y and x == 0 for x, y in zip(map(lambda x: 1 if x!=0 else 0, pruned_model_edges[i].flatten()), map(lambda x: 1 if x!=0 else 0, padawann_edges[i].flatten())))
        sys.stdout.write("\nThe PGA-NN and the PADAWANN have pruned %d edges in common" % same)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PADAWANN-GA", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-im", metavar="Initial Model", type=str, help="the path to the file that contains the initial model's structure and weight and bias values")
    parser.add_argument("-p", metavar="PADAWANN", type=str, help="the path to the file that contains the padawann's structure and weight and bias values")
    parser.add_argument("-ds", metavar="Datasets", type=str, help="the datasets the network will be tested on;\n  \
                                                                one of them MUST be the dataset the initial model was pretrained on\n \
                                                                specified as a comma separated list, e.g. 'mnist,10letters'\n \
                                                                available options: mnist, 10letters, another10, fashion")
    args = parser.parse_args()

    noga = NoGA(args.im, args.p, args.ds.split(","))
    noga.pruning()