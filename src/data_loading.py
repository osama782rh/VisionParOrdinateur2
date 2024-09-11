import os
import cv2


def load_images(folder_path):
    """
    Charge les images depuis le dossier et retourne une liste des images.
    """
    image_list = []

    # Parcourir les sous-dossiers qui contiennent les images
    for label_folder in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_folder)

        # Vérifie si c'est un dossier
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                # Charger l'image
                image = cv2.imread(image_path)
                if image is not None:
                    image_list.append(image)

    return image_list


def get_image_info(images):
    """
    Retourne des informations générales sur les images :
    a. Nombre d'images
    b. Format et taille des images (première image)
    """
    num_images = len(images)
    image_shape = images[0].shape if num_images > 0 else None
    return num_images, image_shape


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_folder = os.path.join(current_dir, '../img', 'data1', 'computer_vision_tp1', 'data1')

    images = load_images(data_folder)

    num_images, image_shape = get_image_info(images)

    print(f"Nombre d'images : {num_images}")
    if image_shape:
        print(f"Format de la première image : {image_shape}")
    else:
        print("Aucune image trouvée.")
