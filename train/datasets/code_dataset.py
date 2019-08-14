from random import shuffle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.util import construct, try_cuda


class CodeDataset(data.Dataset):
    """
    PyTorch dataset that groups samples from an underlying dataset together so
    that they are ready for encoding.
    """

    def __init__(self, name, base_model, num_classes, base_dataset,
                 ec_k, code_dataset=None, put_gpu=True):
        """
        Parameters
        ----------
        name: str
            One of {"train", "val", "test"}
        base_model: ``torch.nn.Module``
            Base model on which inference is being performed and over which a
            code imparts resilience.
        num_classes: int
            The number of classes in the underlying dataset.
        base_dataset: ``torchvision.datasets.Dataset``
            A dataset from the datasets provided by torchvision.
        ec_k: int
            Number of samples from ``base_dataset`` that will be encoded
            together.
        code_dataset: ``torchvision.dataset.Dataset``
            Dataset containing a set of transforms to apply to samples prior to
            encoding. These transforms may differ from those in
            `base_transform` as one may wish to include transformations such as
            random cropping and rotating of images so as to reduce overfiting.
            Such transformations would not be included in `base_dataset` as
            they could lead to noisy labels being generated.
        put_gpu: bool
            Whether to put data and labels on GPU. This is untenable for large
            datasets.
        """
        self.name = name
        self.base_model = base_model
        self.ec_k = ec_k
        self.dataset = base_dataset

        # Since we are not directly calling this DataLoader when we perform
        # iterations when training a code, it is OK not to shuffle the
        # underlying dataset.
        dataloader = data.DataLoader(self.dataset, batch_size=32,
                                     shuffle=False)

        in_size = self.dataset[0][0].view(-1).size(0)
        self.num_channels = self.dataset[0][0].size(0)
        if self.num_channels > 1:
            assert self.num_channels == 3, "Only currently support 3 channels for multi-channel input"

        # Preprate data, outputs from base model, and the true labels for
        # samples. We will populate these tensors so that we can later access
        # them without pulling PIL images from the underlying dataset.
        self.data = torch.zeros(len(self.dataset), in_size)
        self.outputs = torch.zeros(len(self.dataset), num_classes)
        self.true_labels = torch.zeros(len(self.dataset))

        cur = 0
        for inputs, targets in dataloader:
            inputs = try_cuda(inputs.squeeze(1).view(inputs.size(0), -1))
            x = self.base_model(inputs)
            last = cur + inputs.size(0)
            self.data[cur:last, :] = inputs.data
            self.outputs[cur:last, :] = x.data
            self.true_labels[cur:last] = targets
            cur = last

        # Calculate the accuracy of the base model with respect to this dataset.
        base_model_preds = torch.max(self.outputs, dim=1)[1]
        correct_preds = (base_model_preds ==
                         self.true_labels.long())
        base_model_num_correct = torch.sum(correct_preds).item()
        base_model_num_tried = self.outputs.size(0)
        base_model_accuracy = base_model_num_correct / base_model_num_tried
        print("Base model", name, "accuracy is", base_model_num_correct,
              "/", base_model_num_tried, "=", base_model_accuracy)

        self.true_labels = self.true_labels.long()
        if put_gpu:
            # Move data, outputs, and true labels to GPU for fast access.
            self.data = try_cuda(self.data)
            self.outputs = try_cuda(self.outputs)
            self.true_labels = try_cuda(self.true_labels)

        # If extra transformations are passed, create a new dataset containing
        # these so that a caller can pull new, transformed samples with calls
        # to `__getitem__`.
        if name == "train" and code_dataset is not None:
            self.dataset = code_dataset
            self.extra_transforms = True
        else:
            self.extra_transforms = False

    def __getitem__(self, idx):
        # If there are extra transformations to perform, we pull directly from
        # the underlying dataset rather than from the cached `data` tensor
        # because we'd like a new sample, and extra transformations often
        # contain some random components.
        #
        # Note, however, that even though we are pulling a "new" sample from
        # the underlying dataset, we will still use thes same output for the
        # sample as we calculated when we initially performed inference to get
        # the `outputs` tensor during `__init__`. This avoids having to perform
        # inference over the base model in-line with `__getitem__` calls.
        if self.extra_transforms:
            data, _ = self.dataset[idx]
            data = data.view(-1)
        else:
            data = self.data[idx]

        return data, self.outputs[idx], self.true_labels[idx]

    def __len__(self):
        # Number of samples in an epoch is equal to the number of `ec_k`-sized
        # groups are contained in our dataset.
        return (self.data.size(0) // self.ec_k) * self.ec_k

    def encoder_in_dim(self):
        """
        Returns dimensionality of input that will be given to the encoder.
        """
        return self.data.size(1) // self.num_channels

    def decoder_in_dim(self):
        """
        Returns dimensionality of input that will be given to the decoder.
        """
        return self.outputs.size(1)


class DownloadCodeDataset(CodeDataset):
    """
    Wrapper class around CodeDataset for handling datasets made available
    through torchvision.
    """

    def __init__(self, name, base_model, num_classes, base_dataset,
                 base_dataset_dir, ec_k, base_transform=None,
                 code_transform=None):
        """
        Parameters (that are different from CodeDataset)
        ------------------------------------------------
        base_dataset_dir: str
            Location where ``base_dataset`` has been or will be saved. This
            avoids re-downloading the dataset.
        base_transform: ``torchvision.transforms.Transform``
            Set of transforms to apply to samples when generating base model
            outputs that will (potentially) be used as labels.
        """
        if base_transform is None:
            base_transform = transforms.ToTensor()

        # Draw from the torchvisions "train" datasets for training and
        # validation datasets
        is_train = (name != "test")

        # Create the datasets from the underlying `base_model_dataset`.
        # When generating outputs from running samples through the base model,
        # we do apply `base_transform`.
        full_base_dataset = base_dataset(root=base_dataset_dir, train=is_train,
                                         download=True, transform=base_transform)

        if code_transform is not None:
            full_code_dataset = base_dataset(root=base_dataset_dir, train=is_train,
                                             download=True, transform=code_transform)
        else:
            full_code_dataset = None

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=full_base_dataset,
                         ec_k=ec_k,
                         num_classes=num_classes,
                         code_dataset=full_code_dataset)


class FolderCodeDataset(CodeDataset):
    """
    Wrapper class around CodeDataset for handling datasets downloaded on our
    own.
    """

    def __init__(self, name, base_model, num_classes, base_dataset_dir,
                 ec_k, base_transform=None, code_transform=None):
        """
        Parameters (that are different from CodeDataset)
        ------------------------------------------------
        base_dataset_dir: str
            Location where ``base_dataset`` has been or will be saved. This
            avoids re-downloading the dataset.
        base_transform: ``torchvision.transforms.Transform``
            Set of transforms to apply to samples when generating base model
            outputs that will (potentially) be used as labels.
        """
        if base_transform is None:
            base_transform = transforms.ToTensor()

        full_base_dataset = datasets.ImageFolder(
            root=base_dataset_dir, transform=base_transform)

        if code_transform is not None:
            full_code_dataset = datasets.ImageFolder(
                root=base_dataset_dir, transform=code_transform)
        else:
            full_code_dataset = None

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=full_base_dataset,
                         ec_k=ec_k,
                         num_classes=num_classes,
                         code_dataset=full_code_dataset,
                         put_gpu=False)


class MNISTCodeDataset(DownloadCodeDataset):
    def __init__(self, name, base_model, ec_k):
        base_dataset = datasets.MNIST
        base_dataset_dir = "data/mnist"
        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, num_classes=10)


class FashionMNISTCodeDataset(DownloadCodeDataset):
    def __init__(self, name, base_model, ec_k):
        base_dataset = datasets.FashionMNIST
        base_dataset_dir = "data/fashion-mnist"
        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, num_classes=10)


class CIFARCodeDataset(DownloadCodeDataset):
    def __init__(self, name, base_model, ec_k, num_classes):
        assert num_classes == 10 or num_classes == 100

        if num_classes == 10:
            base_dataset = datasets.CIFAR10
            base_dataset_dir = "data/cifar10"
        else:
            base_dataset = datasets.CIFAR100
            base_dataset_dir = "data/cifar100"

        # For `base_transform`, we only apply normalization.
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))])

        # We add extra transformations for CIFAR-10 as is done in:
        #   https://github.com/kuangliu/pytorch-cifar
        code_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010))])

        super().__init__(name=name,
                         base_model=base_model,
                         base_dataset=base_dataset,
                         base_dataset_dir=base_dataset_dir,
                         ec_k=ec_k, num_classes=num_classes,
                         base_transform=base_transform,
                         code_transform=code_transform)


class CIFAR10CodeDataset(CIFARCodeDataset):
    def __init__(self, name, base_model, ec_k):
        super().__init__(name=name, base_model=base_model,
                         ec_k=ec_k, num_classes=10)


class CIFAR100CodeDataset(CIFARCodeDataset):
    def __init__(self, name, base_model, ec_k):
        super().__init__(name=name, base_model=base_model,
                         ec_k=ec_k, num_classes=100)


class CatDogCodeDataset(FolderCodeDataset):
    def __init__(self, name, base_model, ec_k):
        dataset_dir = "data/cat_v_dog/{}"
        base_dataset_dir = dataset_dir.format(name)
        num_classes = 2

        # See: https://github.com/pytorch/examples/blob/master/imagenet/main.py
        base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        if name == "train":
            code_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        else:
            code_transform = None

        super().__init__(name=name, base_model=base_model, ec_k=ec_k,
                         base_dataset_dir=base_dataset_dir,
                         num_classes=num_classes,
                         base_transform=base_transform,
                         code_transform=code_transform)


def get_dataloaders(dataset_path, base_model, ec_k, batch_size):
    """
    Generates training, validation, and test datasets.

    Parameters
    ----------
    dataset_path: str
        Classpath of underlying dataset to use.
    base_model: ``torch.nn.Module``
        Base model on which inference is being performed and over which a
        code imparts resilience.
    ec_k: int
        Number of samples from ``base_dataset`` that will be encoded
        together.
    batch_size: int
        Number of samples (group of `ec_k` inputs) to be run in a single
        minibatch.

    Returns
    -------
    {train, val, test}_dataloader: ``torch.utils.data.DataLoader``
        Dataloaders to be used for training, validation, and testing.
    """
    train_dataset = construct(dataset_path,
                              {"name": "train",
                               "base_model": base_model,
                               "ec_k": ec_k})

    val_dataset = construct(dataset_path,
                            {"name": "val",
                             "base_model": base_model,
                             "ec_k": ec_k})

    test_dataset = construct(dataset_path,
                             {"name": "test",
                              "base_model": base_model,
                              "ec_k": ec_k})

    # Each sample for the encoder/decoder consists of `ec_k` images from
    # the underlying dataset. Thus, the batch size for drawing samples from
    # the underlying dataset is `batch_size * ec_k`
    batch_size_for_loading = ec_k * batch_size
    if "CatDog" in dataset_path["class"] or  "GCommand" in dataset_path["class"]:
        num_workers = 4
        pin_mem = torch.cuda.is_available()
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=batch_size_for_loading,
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=pin_mem)
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=batch_size_for_loading,
                                     shuffle=True, num_workers=num_workers,
                                     pin_memory=pin_mem)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=batch_size_for_loading,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=pin_mem)

    else:
        total_train = len(train_dataset)
        indices = list(range(total_train))
        shuffle(indices)

        num_val = 5000
        remainder = num_val % ec_k
        # Make sure that the training and validation sets have a multiple of
        # ec_k.
        if remainder != 0:
            num_val += (ec_k - remainder)

        train_indices = indices[num_val:]
        val_indices = indices[:num_val]
        train_sampler = data.sampler.SubsetRandomSampler(train_indices)
        val_sampler = data.sampler.SubsetRandomSampler(val_indices)

        train_loader = data.DataLoader(train_dataset, sampler=train_sampler,
                                       batch_size=batch_size_for_loading)

        val_loader = data.DataLoader(val_dataset, sampler=val_sampler,
                                     batch_size=batch_size_for_loading)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=batch_size_for_loading, shuffle=False)
    return train_loader, val_loader, test_loader
