from sklearn.metrics import confusion_matrix
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

# Modèle 2 : SVM
clf_svm = SVC(kernel='linear', random_state=42)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_val)

# Matrice de confusion pour le modèle 1 (Arbre de décision)
confusion_tree = confusion_matrix(y_val, y_pred_tree)
print(f"Matrice de confusion du modèle 1 (Arbre de décision) :\n{confusion_tree}")

# Interprétation :
# confusion_matrix[0, 0] = nombre de bike correctement classifiés comme bike
# confusion_matrix[0, 1] = nombre de bike classifiés comme car
# confusion_matrix[1, 0] = nombre de car classifiés comme bike
# confusion_matrix[1, 1] = nombre de car correctement classifiés comme car
bike_classified_as_car = confusion_tree[0, 1]
car_classified_as_bike = confusion_tree[1, 0]
print(f"Nombre de bike classifiés comme car : {bike_classified_as_car}")
print(f"Nombre de car classifiés comme bike : {car_classified_as_bike}")

# Matrice de confusion pour le modèle 2 (SVM)
confusion_svm = confusion_matrix(y_val, y_pred_svm)
print(f"Matrice de confusion du modèle 2 (SVM) :\n{confusion_svm}")
