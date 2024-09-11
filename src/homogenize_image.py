import os
import cv2
import numpy as np


def load_and_preprocess_images(folder_path, image_size=(224, 224)):
    """
    Charge les images depuis le dossier, redimensionne les images à la taille spécifiée
    et retourne deux tableaux numpy : un pour les images et un pour les labels.
    """
    images = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)

                if image is not None:
                    # Redimensionner l'image à (224, 224)
                    image = cv2.resize(image, image_size)
                    images.append(image)
                    labels.append(label_folder)

    return np.array(images), np.array(labels)


if __name__ == "__main__":
    # Chemin vers le dossier contenant les images
    current_dir = os.path.dirname(__file__)
    data_folder = os.path.join(current_dir, '..', 'img', 'data1', 'computer_vision_tp1', 'data1')

    # Charger et redimensionner les images et obtenir les labels
    images, labels = load_and_preprocess_images(data_folder)

    # Afficher les dimensions des arrays
    print(f"Dimensions des images : {images.shape}")
    print(f"Labels : {labels[:5]}")  # Afficher les 5 premiers labels pour vérification
