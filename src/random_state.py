from sklearn.model_selection import train_test_split
import numpy as np


def split_train_val(image_vectors, labels, test_size=0.2, random_state=42):
    """
    Sépare les données en sets d'entraînement et de validation.
    test_size : proportion des données réservée pour la validation (ici 20%).
    random_state : assure la reproductibilité des résultats.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        image_vectors, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    # Simulons des vecteurs d'images (100 images de 150528 features) et leurs labels
    image_vectors = np.random.rand(100, 150528)  # 100 images, 150528 features chacune
    labels = np.random.choice(['bike', 'car'], 100)  # 100 labels (bike ou car)

    # Séparation en sets d'entraînement et de validation
    X_train, X_val, y_train, y_val = split_train_val(image_vectors, labels)

    print(f"Entraînement : {X_train.shape[0]} images, Validation : {X_val.shape[0]} images")
    print(f"Labels d'entraînement (5 premiers) : {y_train[:5]}")
    print(f"Labels de validation (5 premiers) : {y_val[:5]}")
