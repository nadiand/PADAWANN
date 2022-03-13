# General imports
import numpy as np
import copy
import torch
import sys
import time
import argparse

# Custom imports
from Net import Net
from InitialModel import InitialModel
from DatasetManager import DatasetManager
from Pruner import pruning_mutation
from GAAnalyzer import GAAnalyzer
from Evaluate import evaluate_model

class GA():
    """
    The class representing the genetic algorithm. It has the following attributes:
    * population_size       - the number of networks in a population,
    * iterations            - the number of iterations the GA will take,
    * tournament_size       - the number of individuals that compete in a tournament,
    * elitism               - the portion of the population that gets copied to the next generation 
                              without modification (elites),
    * culling               - the portion of the population that gets killed off (the worst networks),
    * crossover_probability - the probability that crossover will occur for creating a new individual,
    * pruning_amount        - the portion of edges that will be pruned in each evolutionary step,

    * initial_model         - the initial model's structure,
    * population            - a list of networks in the current generation,
    * best_model            - the best model achieved so far according to the fitness measure,

    * dataset_manager       - a data structure containing the datasets used for testing,
    * progress_analyzer     - a data structure facilitating recording the progress of the PADAWANN,
    * padawann_id           - a unique identifier to include in the name of the file in which the 
                              PADAWANN is stored.
    """
    def __init__(self, initial_model="", datasets=[], population_size=10, iterations=50, tournament_size=2,
                    elitism=0.3, culling=0.3, crossover_probability=0.5, pruning_amount=0.025, file_id=""):
        # GA parameters
        self.population_size = population_size
        self.iterations = iterations
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.culling = culling
        self.crossover_probability = crossover_probability
        self.pruning_amount = pruning_amount

        # Initial pre-trained model
        pretrained_model = InitialModel()
        pretrained_model.load_state_dict(torch.load(initial_model))
        self.initial_model = Net(pretrained_model)

        # Datasets for evaluation
        self.dataset_manager = DatasetManager()
        self.dataset_manager.load_datasets(datasets, 'evolution')

        # Population
        self.population = []
        self.best_model = None

        # Facilitated GA analysis
        self.progress_analyzer = GAAnalyzer(datasets)
        self.padawann_id = file_id
        
    def evolve(self):
        """
        Execute the GA with the stopping criterion being surpassing *self.iterations* many iterations. 
        Save the PADAWANN to a file and log its progress.
        """
        start_time = time.time()
        # Create initial population
        self.initialize_population()
        self.progress_analyzer.update(self.best_model)
        # Perform evolution
        max_iterations = self.iterations
        while self.iterations > 0:
            sys.stdout.write("\rGeneration %(gen)d/%(max)d" % {"gen": max_iterations-self.iterations+1, "max": max_iterations})
            self.evolutionary_step()
            self.progress_analyzer.update(self.best_model)
            self.iterations -= 1
        sys.stdout.write("\nDone!")

        # Display and store progress
        self.progress_analyzer.result_report()
        sys.stdout.write("\n----- The GA took %f hours -----\n" % float((time.time()-start_time)/3600.0))
        self.progress_analyzer.store(self.padawann_id)
        
        # Save the best model into a file
        torch.save(self.best_model.model.state_dict(), "padawann_" + self.padawann_id + ".pth")

    def initialize_population(self):
        """
        Create the initial population out of the initial model *self.initial_model* by copying it
        *self.population_size* many times and pruning each copy.
        """
        sys.stdout.write("Creating initial population...")
        # Add initial model to population
        self.calculate_fitness(self.initial_model)
        self.population.append(self.initial_model.make_copy())
        self.progress_analyzer.update(self.initial_model)

        # Add pruned copies of initial model
        while len(self.population) < self.population_size:
            new_individual = self.initial_model.make_copy()
            self.mutate(new_individual)
            self.calculate_fitness(new_individual)
            self.population.append(new_individual.make_copy())
        
        # Initial best model
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_model = self.population[0].make_copy()

        sys.stdout.write("Done!\n")

    def evolutionary_step(self):
        """
        A single evolutionary step of the algorithm: Elitism, culling, tournament selection, 
        crossover, and mutation are performed to create the next population out of the current one.
        """
        # Elitism
        elite = self.population[:int(self.population_size*self.elitism)]
        new_population = [model.make_copy() for model in elite] #RENAME TODO next_population
            
        # Culling
        self.population = self.population[:int(self.population_size*(1-self.culling))]
        
        # Creating the new population
        while len(new_population) < self.population_size:
            new_individual = None
            individual1 = self.tournament_selection(self.population)
            if np.random.uniform(0,1) < self.crossover_probability:
                individual2 = self.tournament_selection(self.population)
                new_individual = self.crossover(individual1.model, individual2.model)
            else:
                new_individual = individual1.make_copy()
            self.mutate(new_individual)
            self.calculate_fitness(new_individual)
            new_population.append(new_individual.make_copy())

        # Sort population and update best model if possible
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        if new_population[0].fitness > self.best_model.fitness:
            self.best_model = new_population[0].make_copy()

        self.population = new_population

    def tournament_selection(self, population):
        """
        Performs tournament selection between *self.tournament_size* many random networks from *population*
        by comparing their fitness and returning the fittest one. 
        """
        # Pick competitors
        competitors = []
        while len(competitors) < self.tournament_size:
            index = np.random.randint(0, len(population))
            competitors.append(population[index].make_copy())
        
        # Find best one and return it
        competitors.sort(key=lambda x: x.fitness, reverse=True)
        return competitors[0].make_copy()
        
    def calculate_fitness(self, individual):
        """
        Calculates the fitness of *individual* by evaluating it on all datasets from the 
        *self.dataset_manager* in a weight-agnostic way. Also updates the attributes of *individual*.
        """
        values = [0.5, 1.0, 2.0]
        accuracies, indices, loss = evaluate_model(individual, self.dataset_manager, values, purpose='evolution')
        individual.loss = loss
        individual.accuracies = accuracies
        individual.weight_vals = [values[ind] for ind in indices]
        individual.fitness = float(sum(individual.accuracies)/sum([100-acc for acc in individual.accuracies]))
    
    def mutate(self, individual):
        """
        Perform a single pruning step on the network *individual*: Using unstructured pruning, randomly 
        prune *self.pruning_amount*% edges.
        """
        # Make a list of all prune-able layers
        layers = []
        for layer in individual.get_all_layers():
            if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Conv3d):
                layers.append(layer)
        # Prune
        pruning_mutation("ru", layers, self.pruning_amount)

    def crossover(self, model1, model2):
        """
        Perform one-point crossover between *model1* and *model2* and return the new network.
        """
        new_model = copy.deepcopy(model1)
        copy_model2 = copy.deepcopy(model2)
        # Randomly pick the crossover point and perform the crossover 
        num_layers = len([module for module in model1.children()])
        point = np.random.randint(0, num_layers-1)
        for ind, (layer1, layer2) in enumerate(zip(new_model.children(), copy_model2.children())):
            if ind >= point:
                layer1 = layer2
        return Net(new_model)

def run(args):
    ga = GA(initial_model=args.im, population_size=args.ps, iterations=args.iter, tournament_size=args.tour, elitism=args.el, culling=args.cul, datasets=args.ds.split(","), crossover_probability=args.co, pruning_amount=args.pa, file_id=args.fid)
    ga.evolve()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PADAWANN-GA", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-im", metavar="initial model", type=str, help="the path to the file that contains the initial model's structure and weight and bias values")
    parser.add_argument("-iter", metavar="iterations", type=int, help="the number of iterations the GA runs for before terminating", default=50)
    parser.add_argument("-ps", metavar="population size", type=int, help="the number of individuals in the population", default=10)
    parser.add_argument("-tour", metavar="tournament size", type=int, help="the number of individuals that compete in a tournament", default=2)
    parser.add_argument("-el", metavar="elitism", type=float, help="the percentage of elites that are copied to the next generation", default=0.3)
    parser.add_argument("-cul", metavar="culling", type=float, help="the percentage of worst networks that get killed off each generation", default=0.3)
    parser.add_argument("-ds", metavar="datasets", type=str, help="the datasets the network will be tested on;\n  \
                                                                one of them should be the dataset the initial model was pretrained on;\n \
                                                                specified as a comma separated list, e.g. 'mnist,10letters';\n \
                                                                available options: mnist, 10letters, another10, fashion")
    parser.add_argument("-fid", metavar="output file identifier", type=str, help="a way to identify the output file that the PADAWANN is stored in\n \
                                                                e.g. a number (ID), special name, etc. as a string")
    parser.add_argument("-pa", metavar="pruning amount", type=float, help="the amount of edges to be pruned in each generation (in %)", default=0.025)
    parser.add_argument("-co", metavar="crossover probability", type=float, help="the probability of there being crossover between two individuals", default=0.5)
    args = parser.parse_args()
    run(args)