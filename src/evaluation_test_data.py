import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# Fonction pour charger et homogénéiser les images (redimensionner à 224x224)
def load_and_preprocess_images(folder_path, image_size=(224, 224)):
    images = []
    labels = []

    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)

                if image is not None:
                    image = cv2.resize(image, image_size)
                    images.append(image)
                    labels.append(label_folder)  # Utilise le nom du dossier comme label (ex: 'bike', 'car')
                else:
                    print(f"Erreur lors du chargement de l'image : {image_path}")

    return np.array(images), np.array(labels)


# a. Charger les données contenues dans test_data1
current_dir = os.path.dirname(__file__)
test_data_folder = os.path.join(current_dir, '..', 'img', 'test_data1', 'computer_vision_tp1', 'val')

# b. Prétraitement des données de test
test_images, test_labels = load_and_preprocess_images(test_data_folder)

if test_images.size == 0:
    print("Aucune image n'a été chargée. Vérifiez le chemin des données de test.")
else:
    # Aplatir les images pour en faire des vecteurs
    test_images_flatten = test_images.reshape(len(test_images), -1)  # Redimensionnement en 2D (nb_images, nb_features)

    # Simuler des données d'entraînement pour avoir un modèle déjà entraîné
    image_vectors = np.random.rand(100, 150528)
    labels = np.random.choice([0, 1], 100)

    # Séparation des données en sets d'entraînement et de validation (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(image_vectors, labels, test_size=0.2, random_state=42)

    # Entraîner un modèle d'arbre de décision (modèle 1)
    clf_tree = DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train, y_train)

    # Entraîner un modèle SVM (modèle 2)
    clf_svm = SVC(kernel='linear', random_state=42)
    clf_svm.fit(X_train, y_train)

    # Choisir le modèle avec la meilleure performance sur les données de validation
    accuracy_tree = accuracy_score(y_val, clf_tree.predict(X_val))
    accuracy_svm = accuracy_score(y_val, clf_svm.predict(X_val))

    if accuracy_tree > accuracy_svm:
        best_model = clf_tree
        print("Le modèle avec la meilleure performance est l'Arbre de décision")
    else:
        best_model = clf_svm
        print("Le modèle avec la meilleure performance est le SVM")

    # c. Calculer l'accuracy de classification des données de test avec le meilleur modèle
    y_pred_test = best_model.predict(test_images_flatten)
    accuracy_test = accuracy_score(test_labels, y_pred_test)
    print(f"Accuracy sur les données de test : {accuracy_test:.2f}")

    # d. Que peut-on dire de cette valeur ?
    if accuracy_test < 0.7:
        print("L'accuracy est faible. Le modèle n'a pas bien généralisé sur les données de test.")
    else:
        print("L'accuracy est satisfaisante. Le modèle a bien généralisé sur les données de test.")

    # e. Comment peut-on l'expliquer ? Régler le problème
    if accuracy_test < 0.7:
        print(
            "Cela peut s'expliquer par une différence de distribution entre les données d'entraînement et les données de test.")
        print("Pour régler ce problème, nous pourrions :")
        print("- Augmenter la taille des données d'entraînement.")
        print("- Utiliser des techniques d'augmentation de données pour mieux représenter les classes.")
        print("- Améliorer le prétraitement des images, comme la normalisation ou l'égalisation d'histogramme.")
