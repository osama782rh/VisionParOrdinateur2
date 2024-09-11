import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Fonction pour charger toutes les images d'un dossier et les transformer en vecteurs
def load_images(folder_path, image_size=(224, 224)):
    images = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)

                if image is not None:
                    image = cv2.resize(image, image_size)
                    images.append(image)

    return np.array(images)


# 1. Charger toutes les images dans un seul dataset
def gather_all_images():
    current_dir = os.path.dirname(__file__)

    # Charger les images d'entraînement, de validation et de test
    train_folder = os.path.join(current_dir, '..', 'img', 'data1', 'computer_vision_tp1', 'data1')
    test_folder = os.path.join(current_dir, '..', 'img', 'test_data1', 'computer_vision_tp1', 'val')

    train_images = load_images(train_folder)
    test_images = load_images(test_folder)

    # Combiner toutes les images
    all_images = np.concatenate([train_images, test_images])

    # Aplatir les images en vecteurs
    all_images_flatten = all_images.reshape(len(all_images), -1)

    return all_images_flatten


# 2. Entraîner un modèle K-means pour 2 clusters
def train_kmeans(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans


# 3. Visualiser les clusters en 2D en utilisant PCA
def visualize_clusters(kmeans, data):
    # Réduction de dimensions avec PCA (de nb_features -> 2 dimensions)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    # Visualiser les clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title('Visualisation des clusters K-means (2D)')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.colorbar(label='Cluster')
    plt.show()

    # Sauvegarder l'image de la figure pour le README
    plt.savefig('img/kmeans_clusters.png')


if __name__ == "__main__":
    # 1. Rassembler toutes les images en un seul dataset
    all_images_flatten = gather_all_images()

    # 2. Entraîner K-means
    kmeans = train_kmeans(all_images_flatten, n_clusters=2)

    # 3. Visualiser les clusters
    visualize_clusters(kmeans, all_images_flatten)
