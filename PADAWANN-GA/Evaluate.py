# General imports
import copy

"""
A method that sets all weight values of the network *individual* to have value *value*. It 
modifies *individual* in place.
"""
def set_param_values(individual, value):
    for child in individual.get_all_layers():
        d = copy.deepcopy(child.state_dict())
        for g in d:
            if g == 'weight':
                d[g] = d[g].sign_()*value
        child.load_state_dict(d)

"""
A helper method that takes network *individual* and tests it on the datasets loaded in 
*dataset_manager* using weight-agnosticism (the shared weight values are provided in *values*).
It returns useful information for updating a Net's attributes or printing the accuracy on the 
datasets. If *conf_mat* is True, it also provides information about the true labels and the 
model's outputs for the creation of a confusion matrix.
"""
def evaluate_model(individual, dataset_manager, values, purpose="", conf_mat=False):
    # Bookkeeping
    total_loss = 0
    avg_accuracy = []
    indices = []
    all_labels = []
    all_outputs = []

    # Iterate over all datasets
    for dataset_nr in range(0, dataset_manager.get_number_of_tasks()):
        # Load a dataset
        dataset = dataset_manager.create_loader(dataset_nr, purpose)

        losses = []
        accuracies = []
        labels = []
        outputs = []
        # At each rollout, assign different shared weight value
        for value in values:
            # Set all weights to the same value and evaluate performance
            set_param_values(individual, value)
            loss, accuracy, l, o = individual.evaluate(dataset)
            # Bookkeeping
            losses.append(loss)
            accuracies.append(accuracy)
            labels.append(l)
            outputs.append(o)

        # Find the shared weight value that results in best accuracy
        best_acc = max(accuracies)
        avg_accuracy.append(best_acc)
        best_ind = accuracies.index(best_acc)
        # Bookkeeping
        total_loss += losses[best_ind]
        indices.append(best_ind)
        all_labels.append(labels[best_ind])
        all_outputs.append(outputs[best_ind])
    if not conf_mat:
        return avg_accuracy, indices, total_loss/dataset_manager.get_number_of_tasks()
    return avg_accuracy, indices, total_loss/dataset_manager.get_number_of_tasks(), all_labels, all_outputs