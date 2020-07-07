import struct
import numpy as np
import binascii

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import csv
import torchvision.transforms as transforms
import torchvision.datasets as dset

MAGIC_NUMBER_HEADER_TRAIN_LABELS = b'\x00\x00\x08\x01'
MAGIC_NUMBER_HEADER_TRAIN_IMAGES = b'\x00\x00\x08\x03'

def __check_header__(f, value):
    header = f.read(4) 
    if header != value:
        raise AssertionError("Header not found. (Detected: " + str(header) + ")")

def load_train_labels():
    with open("train-labels-idx1-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_LABELS)
        (records, ) = struct.unpack(">I", f.read(4))
        values = np.zeros((records))

        for i in range(0, records):
            values[i] = f.read(1)[0]
    
        return values.astype(np.float32)

def load_train_images():
    with open("train-images-idx3-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_IMAGES)
        (records, rows, columns ) = struct.unpack(">III", f.read(4 * 3))

        data = f.read()

        return np.frombuffer(data, dtype=np.uint8).reshape((records, rows, columns)).astype(np.float32)

def load_test_labels():
    with open("t10k-labels-idx1-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_LABELS)
        (records, ) = struct.unpack(">I", f.read(4))
        values = np.zeros((records))

        for i in range(0, records):
            values[i] = f.read(1)[0]
    
        return values.astype(np.float32)

def load_test_images():
    with open("t10k-images-idx3-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_IMAGES)
        (records, rows, columns ) = struct.unpack(">III", f.read(4 * 3))

        data = f.read()

        return np.frombuffer(data, dtype=np.uint8).reshape((records, rows, columns)).astype(np.float32)

def train_set():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = dset.ImageFolder(root="mnist_dataset/train", transform=transform)


    return trainset

def unpack():
    os.makedirs("mnist_dataset", exist_ok=True)
    os.makedirs("mnist_dataset/train", exist_ok=True)
    os.makedirs("mnist_dataset/test", exist_ok=True)
    for i in range(0, 9 + 1):
        os.makedirs(f"mnist_dataset/train/class_{i}", exist_ok=True)
        os.makedirs(f"mnist_dataset/test/class_{i}", exist_ok=True)

    __unpack_train_data__()
    __unpack_test_data__()

def __unpack_test_data__():
    train_labels = None
    with open("t10k-labels-idx1-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_LABELS)
        (records, ) = struct.unpack(">I", f.read(4))

        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    with open("t10k-images-idx3-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_IMAGES)
        (records, rows, columns ) = struct.unpack(">III", f.read(4 * 3))

        data = f.read()

        imgs = np.frombuffer(data, dtype=np.uint8).reshape((records, rows, columns))

        for (idx, label, img) in zip(range(0, records), train_labels, imgs):
            out = Image.fromarray(img, "L")
            name = f"mnist_dataset/test/class_{label}/{idx}.png"
            if not os.path.exists(name):
                out.save(name)

def __unpack_train_data__():
    train_labels = None
    with open("train-labels-idx1-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_LABELS)
        (records, ) = struct.unpack(">I", f.read(4))

        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    with open("train-images-idx3-ubyte", "rb") as f:
        __check_header__(f, MAGIC_NUMBER_HEADER_TRAIN_IMAGES)
        (records, rows, columns ) = struct.unpack(">III", f.read(4 * 3))

        data = f.read()

        imgs = np.frombuffer(data, dtype=np.uint8).reshape((records, rows, columns))

        for (idx, label, img) in zip(range(0, records), train_labels, imgs):
            out = Image.fromarray(img, "L")
            name = f"mnist_dataset/train/class_{label}/{idx}.png"
            if not os.path.exists(name):
                out.save(name)




if __name__ == '__main__':
    unpack()