from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Simulons des vecteurs d'images (100 images avec 150528 caractéristiques) et leurs labels
image_vectors = np.random.rand(100, 150528)
labels = np.random.choice([0, 1], 100)

# Séparation des données en sets d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(image_vectors, labels, test_size=0.2, random_state=42)

# Modèle 1 : Arbre de décision
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_val)
y_pred_proba_tree = clf_tree.predict_proba(X_val)[:, 1]  # Probabilité d'appartenir à la classe 1

# Calcul de la précision et du rappel (recall)
precision_tree = precision_score(y_val, y_pred_tree)
recall_tree = recall_score(y_val, y_pred_tree)
print(f"Précision du modèle 1 : {precision_tree:.2f}")
print(f"Rappel du modèle 1 : {recall_tree:.2f}")

# Courbe ROC
fpr, tpr, _ = roc_curve(y_val, y_pred_proba_tree)
roc_auc = roc_auc_score(y_val, y_pred_proba_tree)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC - Modèle 1 (Arbre de décision)')
plt.legend(loc="lower right")
plt.show()
