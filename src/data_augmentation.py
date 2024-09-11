import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Fonction d'augmentation des données
def augment_data(image, image_size=(224, 224)):
    # Image originale redimensionnée
    original = cv2.resize(image, image_size)

    # a. Transformation de cropping (on coupe une partie de l'image)
    cropped = original[16:208, 16:208]  # On coupe un carré de 192x192
    cropped = cv2.resize(cropped, image_size)  # Redimensionner à nouveau à 224x224

    # b. Transformation en noir et blanc
    grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)  # Convertir en 3 canaux pour rester cohérent avec le reste
    grayscale = cv2.resize(grayscale, image_size)

    return original, cropped, grayscale


# Fonction pour charger et augmenter les données
def load_and_augment_images(folder_path, image_size=(224, 224)):
    augmented_images = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)

                if image is not None:
                    # Augmenter les données : original, cropping, noir et blanc
                    original, cropped, grayscale = augment_data(image, image_size)

                    # Ajouter les images augmentées à la liste
                    augmented_images.extend([original, cropped, grayscale])
                    labels.extend([label_folder] * 3)  # Ajouter le label correspondant pour chaque transformation

    return np.array(augmented_images), np.array(labels)


# Chemin vers les images d'entraînement
current_dir = os.path.dirname(__file__)
train_data_folder = os.path.join(current_dir, '..', 'img', 'data1', 'computer_vision_tp1', 'data1')

# Charger et augmenter les données d'entraînement
train_images, train_labels = load_and_augment_images(train_data_folder)

# Aplatir les images pour en faire des vecteurs
train_images_flatten = train_images.reshape(len(train_images), -1)

# Séparation des données en sets d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(train_images_flatten, train_labels, test_size=0.2, random_state=42)

# Entraîner un modèle d'arbre de décision avec les données augmentées
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)

# Charger les données de test (sans augmentation pour évaluation)
test_data_folder = os.path.join(current_dir, '..', 'img', 'test_data1', 'computer_vision_tp1', 'val')
test_images, test_labels = load_and_augment_images(
    test_data_folder)  # Appliquer également l'homogénéisation aux données de test
test_images_flatten = test_images.reshape(len(test_images), -1)

# Prédire et calculer l'accuracy sur les données de test
y_pred_test = clf_tree.predict(test_images_flatten)
accuracy_test = accuracy_score(test_labels, y_pred_test)
print(f"Accuracy sur les données de test après augmentation : {accuracy_test:.2f}")
