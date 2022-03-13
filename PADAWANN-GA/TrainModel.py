# General imports
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Custom imports
from DatasetManager import DatasetManager
from InitialModel import InitialModel

"""
The following two functions are taken from the official PyTorch library's repository,
showing example usage: https://github.com/pytorch/examples/blob/master/mnist/main.py.
"""

def train(model, device, train_loader, optimizer, epoch):
    """
    Train *model* using the training dataset *train_loader*.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    """
    Evaluate *model* on the test dataset *test_loader* and print its accuracy
    and loss.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main(model, dataset, filename):
    """
    The function to call to train *model* which can be an instance of InitialModel, 
    or some other PyTorch Module structure.
    """
    datasetManager = DatasetManager()
    datasetManager.load_datasets(dataset)
    datasetManager.load_training_datasets(dataset)
    train_loader = datasetManager.create_loader(0, train=True)
    test_loader = datasetManager.create_loader(0, train=False)

    optimizer = optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    device = torch.device("cpu")
    for epoch in range(1, 3 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), filename + ".pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PADAWANN-GA", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-ds", metavar="dataset", type=str, help="the datasets the network will be trained on;\n  \
                                                                available options: mnist, 10letters, another10, fashion")
    parser.add_argument("-name", metavar="network file name", type=str, help="the name of the file in which the network structure will be stored")
    args = parser.parse_args()
    model = InitialModel()
    main(model, [args.ds], args.name)