import gzip
import os
import pickle
import struct
import tarfile
import zipfile
import jax.numpy as jnp
import numpy as np
import pandas as pd
from urllib.request import urlretrieve

def download_and_extract_f16():
    
    # Load the dataset
    url = "https://data.4tu.nl/file/b6dc643b-ecc6-437c-8a8a-1681650ec3fe/5414dfdc-6e8d-4208-be6e-fa553de9866f"
    data_dir = "./data/f16/"
    zip_path = os.path.join(data_dir, "F16GVT_Files.zip")
    extracted_folder = os.path.join(data_dir, "F16GVT_Files")
    os.makedirs(data_dir, exist_ok=True)
    
    if os.path.exists(extracted_folder):
        print("Data alread loaded.")
        return extracted_folder
    
    print(f"Downloading data from {url}...")
    urlretrieve(url, zip_path)
    print("Done!")
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Done!")
    
    return extracted_folder


def load_io_data(filepath, header=True):
    """Load the data, filter missing values."""
    df = pd.read_csv(filepath, header=0 if header else None)
    # df = df.dropna(inplace=False)
    return df.values


def load_f16():
    
    # Get the files of interest
    folder = "./data/f16/F16GVT_Files/BenchmarkData/"
    files = [
        "F16Data_FullMSine_Level1.csv",
        "F16Data_FullMSine_Level2_Validation.csv",
        "F16Data_FullMSine_Level3.csv",
        "F16Data_FullMSine_Level4_Validation.csv",
        "F16Data_FullMSine_Level5.csv",
        "F16Data_FullMSine_Level6_Validation.csv",
        "F16Data_FullMSine_Level7.csv"
    ]
    file_paths = [os.path.join(folder, f) for f in files]
    
    # Use dataset 4 for standardisation
    ref_data = load_io_data(file_paths[3])
    mu = jnp.mean(ref_data[:, :5], axis=0)
    sigma = jnp.std(ref_data[:, :5], axis=0)
    
    def process_file(filepath):
        data = load_io_data(filepath)   # Get data
        data = data[:, :5]              # Select relevant columns
        data = (data - mu) / sigma      # Standardization
        return data[:, :2], data[:, 2:] # Split into the 2 inputs and 3 measurements
    
    # Store data with shape (time, batches, ...)
    datasets = [process_file(fp) for fp in file_paths] 
    u_train = jnp.swapaxes(jnp.array([datasets[i][0] for i in [0, 2, 4, 6]]), 1, 0)
    y_train = jnp.swapaxes(jnp.array([datasets[i][1] for i in [0, 2, 4, 6]]), 1, 0)
    u_val = jnp.swapaxes(jnp.array([datasets[i][0] for i in [1, 3, 5]]), 1, 0)
    y_val = jnp.swapaxes(jnp.array([datasets[i][1] for i in [1, 3, 5]]), 1, 0)
    
    return (u_train, y_train), (u_val, y_val)


def _download_file(url, path):
    """Download a file if it isn't already there."""
    if os.path.exists(path):
        return
    print(f"Downloading {url}...")
    urlretrieve(url, path)
    print("Done!")


def download_mnist():
    """Download the raw MNIST idx files."""
    url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    data_dir = "./data/mnist/"
    os.makedirs(data_dir, exist_ok=True)
    for f in files:
        _download_file(url + f, os.path.join(data_dir, f))
    return data_dir


def _read_idx(filepath):
    """Read an IDX file (the MNIST format) into a numpy array."""
    with gzip.open(filepath, "rb") as f:
        magic, ndim = struct.unpack(">HBB", f.read(4))[1:]
        shape = struct.unpack(">" + "I"*ndim, f.read(4*ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist():
    """Load MNIST as (images, labels) with images in [0,1], shape (N, 28, 28)."""
    folder = download_mnist()
    def load(images, labels):
        x = _read_idx(os.path.join(folder, images)).astype(np.float32) / 255.0
        y = _read_idx(os.path.join(folder, labels)).astype(np.int32)
        return x, y
    train = load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")
    test = load("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    return train, test


def download_and_extract_cifar10():
    """Download and extract the CIFAR-10 python batches."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_dir = "./data/cifar10/"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_folder = os.path.join(data_dir, "cifar-10-batches-py")
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(extracted_folder):
        return extracted_folder

    _download_file(url, tar_path)
    print("Extracting files...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print("Done!")
    return extracted_folder


def load_cifar10():
    """Load CIFAR-10 as (images, labels) with images in [0,1], shape (N, 32, 32, 3)."""
    folder = download_and_extract_cifar10()
    def load(files):
        x, y = [], []
        for f in files:
            with open(os.path.join(folder, f), "rb") as fin:
                batch = pickle.load(fin, encoding="bytes")
            x.append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            y.append(np.array(batch[b"labels"]))
        x = np.concatenate(x).astype(np.float32) / 255.0
        return x, np.concatenate(y).astype(np.int32)
    train = load([f"data_batch_{i}" for i in range(1, 6)])
    test = load(["test_batch"])
    return train, test
