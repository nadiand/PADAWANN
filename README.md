# Pruning-Attained Domain- And Weight-Agnostic Neural Networks

This repository presents the code used to produce the results for the creation of the PADAWANNs, described in the paper "Combining Knowledge From Pre-trained Networks And Weight-Agnosticism For Evolving Transferable Models". 

## Dependencies   
The code needs python (it was run on Python 3.9.7) and a few libraries:   
`pip install np`   
`pip install torch`   
`pip install torchvision`   
`pip install matplotlib`   

## Usage   
### Running the GA   
To replicate our experiments and their results, one can run the GA in the following way:   
`python GA.py -iter 50 -ps 20 -tour 8 -im "InitialNNs/mnist_model.pth" -ds "mnist,10letters" -fid "experiment"`   

### Generalization testing   
To evaluate the PADAWANN evolved from MNIST-Net, for instance, one can execute the following command:   
`python Testing.py -im "InitialNNs/mnist_model.pth" -p "../Results/Champions/padawann.pth" -ds "mnist,10letters,another10,fashion" -tl 5`   

### Pruning   
To prune the initial model, then evaluate and compare the pruned models' structure to PADAWANN, run for instance:   
`python NoGA.py -im "InitialNNs/mnist_model.pth" -p "../Results/Champions/padawann.pth" -ds "mnist,10letters"`