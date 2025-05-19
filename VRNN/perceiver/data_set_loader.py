import torchvision
import os
from torchvision import transforms


def cifar_10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)

    label_classes = ('plane', 'car', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_data, test_data, train_data, label_classes, 1024, 3


def mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    train_data = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)

    test_data = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)

    label_classes = ('0', '1', '2', '3',
                     '4', '5', '6', '7', '8', '9')

    return train_data, test_data, train_data, label_classes, 784, 1

def imagenet100(data_path, image_size=224):
    """
    Load ImageNet-100 dataset from a local folder.
    
    Args:
        data_path: Path to the root directory containing train and val folders
        image_size: Size to resize images (default 224)
        
    Returns:
        train_data, test_data, pre_train_data, label_classes, flat_size, num_channels
    """
    # Standard ImageNet transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets using ImageFolder
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    
    train_data = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )
    
    test_data = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=val_transform
    )
    
    # Use training data as pre-training data (or you could use a different subset)
    pre_train_data = train_data
    
    # Get class names from folder names
    label_classes = [c for c, _ in train_data.class_to_idx.items()]
    
    # Calculate flat size based on image dimensions
    flat_size = image_size * image_size * 3
    
    # Return RGB channels
    num_channels = 3
    
    return train_data, test_data, pre_train_data, label_classes, flat_size, num_channels

def stl10(download=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

    train_data = torchvision.datasets.STL10(root='./data', split="train",
                                            download=download, transform=transform)

    test_data = torchvision.datasets.STL10(root='./data', split="test",
                                           download=download, transform=transform)

    pre_train_data = torchvision.datasets.STL10(root='./data', split="unlabeled",
                                                download=download, transform=transform)

    label_classes = ('0', '1', '2', '3',
                     '4', '5', '6', '7', '8', '9')

    return train_data, test_data, pre_train_data, label_classes, 9216, 3