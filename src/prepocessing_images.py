import os
import cv2
import numpy as np


def load_and_preprocess_images_to_vectors(folder_path, image_size=(224, 224)):
    """
    Charge les images depuis le dossier, redimensionne les images à la taille spécifiée,
    et retourne un tableau numpy de taille (nb_images, nb_features).
    Chaque image est représentée par un vecteur.
    """
    image_vectors = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)

                if image is not None:
                    # Redimensionner l'image à la taille spécifiée (224x224)
                    image = cv2.resize(image, image_size)
                    # Aplatir l'image en un vecteur (taille: 224*224*3)
                    image_vector = image.flatten()
                    image_vectors.append(image_vector)
                    labels.append(label_folder)  # Utilise le nom du dossier comme label (ex: 'bike', 'car')

    return np.array(image_vectors), np.array(labels)


if __name__ == "__main__":
    # Chemin vers le dossier contenant les images
    current_dir = os.path.dirname(__file__)
    data_folder = os.path.join(current_dir, '..', 'img', 'data1', 'computer_vision_tp1', 'data1')

    # Charger les images et les transformer en vecteurs
    image_vectors, labels = load_and_preprocess_images_to_vectors(data_folder)

    # Afficher la taille des données
    print(f"Dimensions des vecteurs d'images : {image_vectors.shape}")  # (nb_images, nb_features)
    print(f"Labels : {labels[:5]}")  # Afficher les 5 premiers labels pour vérification
