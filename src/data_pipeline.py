import torch # the main module from pytorch that has everything needed to handle AI tasks.
from datasets import load_dataset # the function from huggingface that allows us to load the different datasets.
import numpy as np # numpy library for math manipulations and fast arrays.

def get_mnist_data() -> tuple[torch.Tensor, torch.Tensor]:
    """"
    Load and preprocesse the MINIST dataset for generative modeling.

    Download the MNIST dataset from Hugging Face, scale pixel values to [0,1], flatten the 28 * 28 images into 784-dimensional vectors,
    and binarize the values using a > 0.5 threshold.

    Returns:
    tuple[torch.FloatTensor, torch.FloatTensor]: A tuple containing:
        - x_train (torch.FloatTensor): Training data tensor of shape (N, 784)
            containing binarized float values.
        - x_test (torch.FloatTensor): Testing data tensor of shape (N, 784)
            containing binarized float values.

    """

    dataset = load_dataset("ylecun/mnist")

    dataset.set_format("torch", columns = ["image"])

    x_train = torch.tensor(np.array(dataset["train"]["image"])).float()
    x_test = torch.tensor(np.array(dataset["test"]["image"])).float()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.view(-1, 784)
    x_test = x_test.view(-1, 784)

    x_train = (x_train > 0.5).float()
    x_test = (x_test > 0.5).float()

    return x_train, x_test



def get_fashion_mnist_data() -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """"
    Load and preprocesse the Fashion MINIST dataset for generative modeling.

    Download the Fashion MNIST dataset from Hugging Face, scale pixel values to [0,1], flatten the 28 * 28 images into 784-dimensional vectors,
    and binarize the values using a > 0.5 threshold.

    Returns:
    tuple[torch.FloatTensor, torch.FloatTensor]: A tuple containing:
        - x_train (torch.FloatTensor): Training data tensor of shape (N, 784)
            containing binarized float values.
        - x_test (torch.FloatTensor): Testing data tensor of shape (N, 784)
            containing binarized float values.

    """



    dataset = load_dataset("zalando-datasets/fashion_mnist")

    dataset.set_format("torch", columns = ["image"])

    x_train = torch.tensor(np.array(dataset["train"]["image"])).float()
    x_test = torch.tensor(np.array(dataset["test"]["image"])).float()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.view(-1, 784)
    x_test = x_test.view(-1, 784)

    x_train = (x_train > 0.5).float()
    x_test = (x_test > 0.5).float()



    return x_train, x_test



def calculate_sparsity_metrics(dataset_tensor: torch.FloatTensor) -> tuple[float, float]:
    """Calculates both the density (fraction of 1s) and sparsity (fraction of 0s).
    
    Args:
        dataset_tensor (torch.FloatTensor): A binary PyTorch float tensor.
            
    Returns:
        tuple[float, float]: A tuple containing (density, sparsity).
    """
    
    density = dataset_tensor.mean().item()
    sparsity = 1.0 - density
    return round(density, 4), round(sparsity, 4)