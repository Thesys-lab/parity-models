# Datasets
This directory contains PyTorch `Dataset`s that were used in the experiments
in the paper. Specifically, the directory contains `Dataset`s for the
MNIST (`MNISTCodeDataset`), Fashion-MNIST (`FashionMNISTCodeDataset`), and CIFAR-10
(`CIFAR10CodeDataset`) datasets.

### Adding new datasets
The datasets currently supported make use of `CodeDataset`, a wrapper around
datasets that come as part of the `torchvision` package.

If you would like to add a dataset that is not part of `torchvision`, you will
need to create a new wrapper dataset similar to `CodeDataset`. This might wrap
around a `torchvision` `DatasetFolder` or `ImageFolder`. Please raise an issue
if you'd like assistance with this, and we'll be happy to help!
