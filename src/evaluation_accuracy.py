from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Simulons des vecteurs d'images (100 images avec 150528 caractéristiques) et leurs labels
image_vectors = np.random.rand(100, 150528)
labels = np.random.choice([0, 1], 100)

# Séparation des données en sets d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(image_vectors, labels, test_size=0.2, random_state=42)

# Modèle 1 : Arbre de décision
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_val)
accuracy_tree = accuracy_score(y_val, y_pred_tree)

# Modèle 2 : SVM
clf_svm = SVC(kernel='linear', random_state=42)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_val)
accuracy_svm = accuracy_score(y_val, y_pred_svm)

print(f"Accuracy du modèle 1 (Arbre de décision) : {accuracy_tree:.2f}")
print(f"Accuracy du modèle 2 (SVM) : {accuracy_svm:.2f}")
