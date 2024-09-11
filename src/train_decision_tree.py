from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Simulons des vecteurs d'images (100 images avec 150528 caractéristiques) et leurs labels
image_vectors = np.random.rand(100, 150528)  # 100 images, 150528 caractéristiques chacune
labels = np.random.choice([0, 1], 100)  # 100 labels binaires (0 ou 1)

# Séparation des données en sets d'entraînement et de validation (80% train, 20% val)
X_train, X_val, y_train, y_val = train_test_split(image_vectors, labels, test_size=0.2, random_state=42)

# Créer un modèle d'arbre de décision
clf = DecisionTreeClassifier(random_state=42)

# Entraîner le modèle avec les données d'entraînement
clf.fit(X_train, y_train)

# Prédire les labels pour les données de validation
y_pred = clf.predict(X_val)

# Calculer l'accuracy de la classification
accuracy = accuracy_score(y_val, y_pred)

# Afficher les résultats
print(f"Accuracy du modèle d'arbre de décision : {accuracy:.2f}")
