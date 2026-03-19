from sklearn.decomposition import PCA
from loguru import logger

import tensorflow as tf
import time
from torchvision import datasets, transforms
import numpy as np

from classic.ModelsMNIST import TripletAutoencoder, train_triplet_autoencoder, extract_embeddings


def data_load_and_process_mnist(
        num_classes,
        all_samples,
        seed,
        num_examples_per_class,
        pca=True,
        n_features=8,
        epochs = 300,
        margin=.2,
        alpha=1.,
):
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    x_train = mnist_train.data.numpy().astype(np.float32) / 255.0
    y_train = mnist_train.targets.numpy()

    x_test = mnist_test.data.numpy().astype(np.float32) / 255.0
    y_test = mnist_test.targets.numpy()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    if not all_samples:
        selected_indices = []

        for class_label in range(10):
            indices = np.where(y_train == class_label)[0][:num_examples_per_class]
            selected_indices.extend(indices)

        x_train_subset = x_train[selected_indices]
        y_train_subset = y_train[selected_indices]

        shuffle_indices = np.random.permutation(len(x_train_subset))
        x_train = x_train_subset[shuffle_indices]
        y_train = y_train_subset[shuffle_indices]

    logger.info("Shape of subset training data: {}", x_train.shape)
    logger.info("Shape of subset training labels: {}", y_train.shape)

    mask_train = np.isin(y_train, range(0, num_classes))
    mask_test = np.isin(y_test, range(0, num_classes))

    X_train = x_train[mask_train].reshape(-1, 784)
    X_test = x_test[mask_test].reshape(-1, 784)

    Y_train = y_train[mask_train]
    Y_test = y_test[mask_test]

    logger.info("Shape of subset training data: {}", X_train.shape)
    logger.info("Shape of subset training labels: {}", Y_train.shape)
    logger.info("Shape of testing data: {}", X_test.shape)
    logger.info("Shape of testing labels: {}", Y_test.shape)
    if pca:
        start = time.time()
        pca = PCA(n_features)
        X_train = pca.fit_transform(X_train)
        end = time.time()
        total_time = end - start
        X_test = pca.transform(X_test)
    else:
        autoencoder = TripletAutoencoder(bottleneck_dim=n_features)
        start = time.time()
        autoencoder = train_triplet_autoencoder(
            autoencoder,
            X_train,
            Y_train,
            n_epochs=epochs,
            batch_size=100,
            lr=1e-3,
            margin=margin,
            alpha=alpha,
        )
        end = time.time()
        total_time = end - start

        X_train = extract_embeddings(autoencoder, X_train)
        X_test = extract_embeddings(autoencoder, X_test)

    X_train = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min()))
    X_test = (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))

    return X_train, X_test, Y_train, Y_test, total_time
