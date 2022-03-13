import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import sys

class DatasetManager():
    """
    The class facilitating work with PyTorch datasets. It has the following attributes:
    * transform          - the transformation (pre-processing) that needs to be done on the datasets,
    * datasets           - a list with the names of all datasets used for evolution/testing/finetuning,
    * available_datasets - a dictionary with the name of every dataset that can be used by the GA, 
                           the method used for loading it, and the class labels.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.datasets = []
        self.available_datasets = {'mnist': ('MNIST', self.mnist, ['0','1','2','3','4','5','6','7','8','9']), \
                                   '10letters': ('EMNIST 10Letters', self.emnist_10letters, ['a','b','c','d','e','f','g','h','i','j']), \
                                   'another10': ('EMNIST Another10Letters', self.emnist_another10, ['k','l','m','n','o', 'p','q','r','s','t']), \
                                   'fashion': ('FashionMNIST', self.fashion_mnist, ['top','pants','pullover','dress','coat','sandal','shirt','shoe','bag','boot'])}

    def load_datasets(self, dataset_list, purpose):
        """
        Loads the datasets given in *dataset_list* into *self.dataset*. *purpose* specifies if the 
        test, a subset of the train or the full train data should be loaded. Possible values are 
        'test', 'evolution' and 'finetuning'.
        """
        for i, d in enumerate(dataset_list):
            sys.stdout.write("%(ds)d: %(name)s\n" % {"ds" : i+1, "name" : self.available_datasets[d][0]})
            self.available_datasets[d][1](purpose)
            if d not in self.datasets:
                self.datasets.append(d)

    def get_class_labels(self, dataset_index):
        """
        Returns the labels of the dataset classes. Useful for plotting the confusion matrix.
        """
        return self.available_datasets[self.datasets[dataset_index]][2]

    def create_loader(self, index, purpose):
        """
        Loads an iterable over the dataset with purpose *purpose*, the name of which is stored
        at *self.datasets[index]*.
        """
        dataset = "Datasets/" + self.datasets[index] + "_" + purpose + ".pth"
        return torch.load(dataset)

    def get_number_of_tasks(self):
        """
        Returns the number of datasets that are currently in use.
        """
        return len(self.datasets)

    def get_dataset_name(self, index):
        """
        Returns the name of the dataset that is at position *self.datasets[index]*.
        """
        return self.available_datasets[self.datasets[index]][0]

    """
    The following methods are for loading the datasets from the PyTorch library. More methods 
    can easily be added in the same manner, to extend the available datasets.

    For the purpose 'evolution' a subset of 700 x 10 = 7000 samples from the training set is 
    taken, in order to speed up the GA runtime.
    """
    
    def mnist(self, purpose):
        mnist = []
        if purpose == "test":
            mnist = datasets.MNIST('/data', train=False, download=True, transform=self.transform)
        elif purpose == "evolution": 
            mnist_full = datasets.MNIST('/data', train=True, download=True, transform=self.transform)
            classes = np.zeros(10)
            for (img, l) in mnist_full:
                if classes[l] < 700:
                    classes[l] += 1
                    mnist.append((img,l))
        else:
            mnist = datasets.MNIST('/data', train=True, download=True, transform=self.transform)
        loader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)
        torch.save(loader, "Datasets/mnist_" + purpose + ".pth")

    def emnist_10letters(self, purpose):
        dataset_EMNIST_10Letters = []
        if purpose == "test":
            dataset_EMNIST = datasets.EMNIST('/data', train=False, download=True, transform=self.transform, split='letters')
            dataset_EMNIST_10Letters = torch.utils.data.Subset(dataset_EMNIST, np.arange(0,7999))
        elif purpose == "evolution":
            dataset_EMNIST = datasets.EMNIST('/data', train=True, download=True, transform=self.transform, split='letters')
            classes = np.zeros(10)
            # The test dataset is shuffled, so we manually look for the samples we want
            for (img, l) in dataset_EMNIST:
                if l >=1 and l <=10:
                    if classes[l-1] < 700:
                        classes[l-1] += 1
                        dataset_EMNIST_10Letters.append((img,l))
        else:
            dataset_EMNIST = datasets.EMNIST('/data', train=True, download=True, transform=self.transform, split='letters')
            for (img, l) in dataset_EMNIST:
                if l >=1 and l <=10:
                    dataset_EMNIST_10Letters.append((img,l))
        dataset_EMNIST_10Letters = [(data,label-1) for (data,label) in dataset_EMNIST_10Letters]
        loader = torch.utils.data.DataLoader(dataset_EMNIST_10Letters, batch_size=100, shuffle=True)
        torch.save(loader, "Datasets/10letters_" + purpose + ".pth")

    def emnist_another10(self, purpose):
        dataset_EMNIST_Another10Letters = []
        if purpose == "test":
            dataset_EMNIST = datasets.EMNIST('/data', train=False, download=True, transform=self.transform, split='letters')
            dataset_EMNIST_Another10Letters = torch.utils.data.Subset(dataset_EMNIST, np.arange(8000,15999))
        elif purpose == "evolution":
            dataset_EMNIST = datasets.EMNIST('/data', train=True, download=True, transform=self.transform, split='letters')
            classes = np.zeros(10)
            # The test dataset is shuffled, so we manually look for the samples we want
            for (img, l) in dataset_EMNIST:
                if l >=11 and l <=20:
                    if classes[l-11] < 700:
                        classes[l-11] += 1
                        dataset_EMNIST_Another10Letters.append((img,l))
        else:
            dataset_EMNIST = datasets.EMNIST('/data', train=True, download=True, transform=self.transform, split='letters')
            classes = np.zeros(10)
            for (img, l) in dataset_EMNIST:
                if l >=11 and l <=20:
                    dataset_EMNIST_Another10Letters.append((img,l))
        dataset_EMNIST_Another10Letters = [(data,label-11) for (data,label) in dataset_EMNIST_Another10Letters]

        loader = torch.utils.data.DataLoader(dataset_EMNIST_Another10Letters, batch_size=100, shuffle=True)
        torch.save(loader, "Datasets/another10_" + purpose + ".pth")

    def fashion_mnist(self, purpose):
        dataset_fashionMNIST = []
        if purpose == "test":
            dataset_fashionMNIST = datasets.FashionMNIST('/data', train=False, download=True, transform=self.transform)
        elif purpose == "evolution":
            dataset_FashionMNIST_full = datasets.FashionMNIST('/data', train=True, download=True, transform=self.transform) 
            classes = np.zeros(10)
            for (img, l) in dataset_FashionMNIST_full:
                if classes[l] < 700:
                    classes[l] += 1
                    dataset_fashionMNIST.append((img,l))
        else:
            dataset_fashionMNIST = datasets.FashionMNIST('/data', train=True, download=True, transform=self.transform) 
        loader = torch.utils.data.DataLoader(dataset_fashionMNIST, batch_size=100, shuffle=True)
        torch.save(loader, "Datasets/fashion_" + purpose + ".pth")