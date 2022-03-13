# General imports
import matplotlib.pyplot as plt
import torch
import argparse
import sys    
import seaborn as sn
from sklearn.metrics import confusion_matrix

# Custom imports
from Net import Net
from InitialModel import InitialModel
from DatasetManager import DatasetManager
from Evaluate import evaluate_model, set_param_values

dataset_manager = DatasetManager()

def retrain_classification_layers(model, dataset):
    """
    Retrains the last layer of *model* using the train dataset *dataset*.
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Make sure only the last layer's weights and biases will be updated
    params_to_update = []
    for name, param in model.named_parameters():
        if not "fc2" in name:
            param.requires_grad = False
        else:
            params_to_update.append(param) 
    optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Fine-tune the model
    for data in dataset:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test_notl(individual, values):
    """
    Calculates the accuracy of *individual* by evaluating it on all datasets loaded in 
    *dataset_manager* using weight-agnosticism, and updates the shared weight values 
    which achieved the best accuracy.
    """
    # Evaluate the network on all datasets
    accuracies, indices, loss, labels, outputs = evaluate_model(individual, dataset_manager, values, "test", True)
    individual.weight_vals = [values[ind] for ind in indices]
    # For each dataset, display accuracy and confusion matrix
    for dataset_nr in range(dataset_manager.get_number_of_tasks()):
        sys.stdout.write("\nDataset #%(ds)d: %(acc)f%%" % {"ds" : dataset_nr+1, "acc" : accuracies[dataset_nr]})
        plot_confusion_matrix(labels[dataset_nr], outputs[dataset_nr], dataset_nr)
    sys.stdout.write("\nThe loss is " + str(loss))

def test_tl(model, dataset, value, reprtitions):
    """
    Calculates the accuracy of *model* by evaluating it on *dataset* using weight-agnosticism 
    (with shared weight value *value*), after fine-tuning the last layer.
    """
    # Set weight values
    set_param_values(model, value)
    avg_accuracy = 0
    for _ in range(reprtitions):
        # Retrain last layer
        retrain_classification_layers(model.model, dataset_manager.create_loader(dataset, "finetuning"))
        # Evaluate the network and plot its confusion matrix
        _, accuracy, labels, outputs = model.evaluate(dataset_manager.create_loader(dataset, "test"))
        avg_accuracy += accuracy
        plot_confusion_matrix(labels, outputs, dataset)
    sys.stdout.write("\nDataset #%(ds)d: %(acc)f%%, achieved with shared weight value %(wv)f" % {"ds" : dataset+1, "acc" : + avg_accuracy/5, "wv" : value})

def plot_confusion_matrix(y_test, y_test_pred, dataset):
    """
    Creates a confusion matrix, displays it and saves it into a file.
    """
    conf_mat = confusion_matrix(y_test, y_test_pred)
    labels = dataset_manager.get_class_labels(dataset)
    plt.figure()
    sn.heatmap(conf_mat, annot=True, square=True, fmt='.0f', vmin=0, vmax=conf_mat.max(), xticklabels=labels, yticklabels=labels)
    plt.title("Confusion matrix for dataset " + dataset_manager.get_dataset_name(dataset))
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing The PADAWANN", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", metavar="PADAWANN", type=str, help="the path to the file that contains the PADAWANN structure and weight and bias values")
    parser.add_argument("-im", metavar="Initial Model", type=str, help="the path to the file that contains the initial model's structure and weight and bias values")
    parser.add_argument("-ds", metavar="Datasets", type=str, help="the datasets the network will be tested on;\n  \
                                                                one of them MUST be the dataset the initial model was pretrained on\n \
                                                                specified as a comma separated list, e.g. 'mnist,10letters'\n \
                                                                available options: mnist, 10letters, another10, fashion")
    parser.add_argument("-tl", metavar="Transfer Learning", type=int, help="the number of trials over which to average the results of tranfer learning\n \
                                                                if no TL should be applied, do not provide this argument", default=0)
    args = parser.parse_args()

    # Load models
    pretrained_model = InitialModel()
    pretrained_model.load_state_dict(torch.load(args.im))
    initial_model = Net(pretrained_model)

    padawann_model = InitialModel()
    padawann_model.load_state_dict(torch.load(args.p))
    padawann = Net(padawann_model)

    # Load datasets
    dataset_manager.load_datasets(args.ds.split(","), "test")
    if args.tl != 0:
        dataset_manager.load_datasets(args.ds.split(","), "finetuning")

    # Evaluation
    values = [0.5, 1.0, 2.0]
    sys.stdout.write("Before transfer learning:")
    test_notl(padawann, values)
    if args.tl != 0:
        sys.stdout.write("\nAfter transfer learning:")
        for dataset, value in enumerate(padawann.weight_vals):
            test_tl(padawann, dataset, value, args.tl)
    sys.stdout.write("\nThe PADAWANN is %f times sparser than the initial model.\n" % (initial_model.get_nonzero_edges()/padawann.get_nonzero_edges()))