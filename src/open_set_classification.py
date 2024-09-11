import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# a. Qu'est-ce que la classification open-set?
# (Voir section README pour la réponse)

# b. Charger et séparer les classes en 'known' et 'unknown'
def load_cifar10_known_unknown():
    classes_known = [0, 1, 2]  # Par exemple : Avion, Voiture, Oiseau
    class_unknown = 3  # Par exemple : Chat

    # Charger le dataset complet
    dataset, info = tfds.load('cifar10', split='train', as_supervised=True, with_info=True)

    # Séparer les données connues et inconnues
    X_known, y_known = [], []
    X_unknown, y_unknown = [], []

    for image, label in tfds.as_numpy(dataset):
        if label in classes_known:
            X_known.append(image)
            y_known.append(label)
        elif label == class_unknown:
            X_unknown.append(image)
            y_unknown.append(label)

    return np.array(X_known), np.array(y_known), np.array(X_unknown), np.array(y_unknown)


# Séparer les sets d'entraînement, validation et test à partir des données "connues"
def split_known_data(X_known, y_known):
    X_train, X_temp, y_train, y_temp = train_test_split(X_known, y_known, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# c. Entraîner un KNN
def train_knn(X_train, y_train, n_neighbors=3):
    X_train_flat = X_train.reshape(len(X_train), -1)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_flat, y_train)
    return knn


# d. Évaluer le modèle et afficher la matrice de confusion
def evaluate_model(knn, X_val, y_val):
    X_val_flat = X_val.reshape(len(X_val), -1)
    y_pred = knn.predict(X_val_flat)

    accuracy_val = accuracy_score(y_val, y_pred)
    confusion_val = confusion_matrix(y_val, y_pred)

    print(f"Accuracy de validation : {accuracy_val:.2f}")
    print(f"Matrice de confusion de validation :\n{confusion_val}")

    return accuracy_val, confusion_val


# f. Évaluer le modèle sur la classe inconnue
def evaluate_unknown(knn, X_unknown, y_unknown):
    X_unknown_flat = X_unknown.reshape(len(X_unknown), -1)
    y_pred_unknown = knn.predict(X_unknown_flat)

    accuracy_unknown = accuracy_score(y_unknown, y_pred_unknown)
    confusion_unknown = confusion_matrix(y_unknown, y_pred_unknown)

    print(f"Score sur les données inconnues (classe 'chat') : {accuracy_unknown:.2f}")
    print(f"Matrice de confusion sur les données inconnues :\n{confusion_unknown}")

    return accuracy_unknown, confusion_unknown


# g. Détection des classes inconnues
def detect_unknown(knn, X_test, threshold=0.8):
    X_test_flat = X_test.reshape(len(X_test), -1)
    probs = knn.kneighbors(X_test_flat, return_distance=False)

    # Détection des exemples dont les plus proches voisins sont à une distance > threshold
    unknown_detection = (probs.mean(axis=1) < threshold).astype(int)

    print(f"Résultats de la détection d'anomalies : {unknown_detection}")
    return unknown_detection


if __name__ == "__main__":
    # b. Charger les données
    X_known, y_known, X_unknown, y_unknown = load_cifar10_known_unknown()

    # Séparer les données "connues" en sets d'entraînement, validation et test
    X_train, X_val, X_test, y_train, y_val, y_test = split_known_data(X_known, y_known)

    # c. Entraîner un modèle KNN
    knn_model = train_knn(X_train, y_train, n_neighbors=3)

    # d. Évaluer le modèle de validation
    accuracy_val, confusion_val = evaluate_model(knn_model, X_val, y_val)

    # f. Évaluer le modèle sur les données inconnues
    accuracy_unknown, confusion_unknown = evaluate_unknown(knn_model, X_unknown, y_unknown)

    # g. Détection d'anomalies sur le set de test
    unknown_detection = detect_unknown(knn_model, X_test, threshold=0.8)
