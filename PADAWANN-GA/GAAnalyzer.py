# General import
import sys

# Custom import
from Net import Net

class GAAnalyzer():
    """
    The class aggregating the best model's parameters to facilitate analysis. It has as attributes
    lists that contain the best model's parameters after each generation, as well as *last_change* 
    which records the epoch in which the best model was found, and *datasets* which contains a 
    list of the names of the datasets used in the GA.
    """
    def __init__(self, datasets):
        self.best_model_accuracies = [[] for _ in range(len(datasets))]
        self.best_model_loss = []
        self.best_model_parameters = []
        self.best_model_fitness = []
        self.last_change = 0
        self.datasets = datasets

    def update(self, model: Net):
        """
        Updates the lists with the parameters of *model*.
        """
        for i, acc in enumerate(model.accuracies):
            self.best_model_accuracies[i].append(acc)
        self.best_model_loss.append(model.loss)
        self.best_model_parameters.append(model.get_nonzero_edges())
        self.best_model_fitness.append(model.fitness)

        if len(self.best_model_fitness) > 1 and self.best_model_fitness[-1] != self.best_model_fitness[-2]:
            self.last_change = len(self.best_model_fitness)

    def result_report(self):
        """
        Prints the details of the PADAWANN - the best model achieved by the GA.
        """
        report = "The PADAWANN has fitness " + str(self.best_model_fitness[-1])
        for i, dataset in enumerate(self.best_model_accuracies):
            report += "\nOn dataset " + str(self.datasets[i]) + " it has " + str(dataset[-1]) + "% accuracy"
        report += "\nIt has " + str(self.best_model_loss[-1]) + " loss and " + str(self.best_model_parameters[-1]) + " edges"
        report += "\nIt was found in epoch " + str(self.last_change-2) # -2 because we also store the initial model's info and the best model from initial population
        sys.stdout.write(report)
        sys.stdout.flush()
    
    def store(self, file_id):
        """
        Saves all information about the evolution of the PADAWANN into a file which is identified
        by the *file_id* in its name.
        """
        f = open("padawann_" + file_id + "_log.txt", "w")
        f.write("Accuracy evolution:")
        for i, dataset in enumerate(self.best_model_accuracies):
            f.write("\nAccuracy on %s:\n" % self.datasets[i])
            for value in dataset:
                f.write(str(value) + ",")
            sys.stdout.flush()
        f.write("\nNumber of parameters:\n")
        for param in self.best_model_parameters:
            f.write(str(param) + ",")
        sys.stdout.flush()
        f.write("\nLoss evolution:\n")
        for loss in self.best_model_loss:
            f.write(str(loss) + ",")
        sys.stdout.flush()
        f.write("\nFitness evolution:\n")
        for fit in self.best_model_fitness:
            f.write(str(fit) + ",")
        f.write("\nNumber of edges removed: %(edges)d and the PADAWANN is %(t)f times sparser" % {"edges" : self.best_model_parameters[0]-self.best_model_parameters[-1], "t" : self.best_model_parameters[0]/self.best_model_parameters[-1]})
        f.close()