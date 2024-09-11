import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 3.c. Installer TensorFlow Datasets (si nécessaire via le terminal)
# pip install tensorflow-datasets

# 3.d. Accéder à une partie du dataset choisi via tensorflow-datasets (CIFAR-10)
def load_cifar10_subset(classes_subset):
    dataset, info = tfds.load('cifar10', split='train', with_info=True, as_supervised=True)

    # Filtrer le dataset pour ne garder que les 4 classes choisies
    filtered_dataset = dataset.filter(lambda img, label: tf.reduce_any([label == cls for cls in classes_subset]))

    return filtered_dataset, info


# 3.e. Préparer une base de données de classification avec 4 classes
def prepare_data_for_classification(dataset, classes_subset):
    X, y = [], []

    for image, label in tfds.as_numpy(dataset):
        if label in classes_subset:
            X.append(image)
            y.append(np.where(classes_subset == label)[0][0])  # Convertir label en index de la classe dans subset

    return np.array(X), np.array(y)


# 3.f. Décrire le dataset
def describe_dataset(X, y, classes_subset):
    num_images = len(X)
    num_classes = len(classes_subset)

    print(f"Nombre d'images : {num_images}")
    print(f"Nombre de classes : {num_classes}")
    print(f"Classes : {classes_subset}")

    # Compter le nombre d'images par classe
    class_counts = [np.sum(y == i) for i in range(num_classes)]
    for i, count in enumerate(class_counts):
        print(f"Classe {classes_subset[i]} : {count} images")

    # Visualiser quelques images
    visualize_samples(X, y, classes_subset)


# Visualiser quelques images
def visualize_samples(X, y, classes_subset):
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[i])
        plt.title(f"Classe : {classes_subset[y[i]]}")
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # 3.a. Lire la documentation de CIFAR-10
    classes_cifar10 = np.array([0, 1, 2, 3])  # 4 classes : avion, auto, oiseau, chat (labels 0, 1, 2, 3)

    # 3.d. Charger un subset de CIFAR-10 avec ces 4 classes
    dataset, info = load_cifar10_subset(classes_cifar10)

    # 3.e. Préparer les données
    X, y = prepare_data_for_classification(dataset, classes_cifar10)

    # 3.f. Décrire le dataset
    describe_dataset(X, y, classes_cifar10)
