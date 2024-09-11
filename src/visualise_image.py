import os
import cv2
import matplotlib.pyplot as plt


def visualize_image(image_path):
    """
    Affiche une image donnée en utilisant matplotlib.
    """
    image = cv2.imread(image_path)
    if image is not None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convertir BGR en RGB
        plt.axis('off')  # Pas d'axes
        plt.show()
    else:
        print("Erreur : Impossible de charger l'image.")


if __name__ == "__main__":
    # Chemin vers une image spécifique dans le dossier 'img/data1/computer_vision_tp1/data1'
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, '..', 'img', 'data1', 'computer_vision_tp1', 'data1', 'bike',
                              'Bike (1).png')

    # Visualiser l'image
    visualize_image(image_path)
