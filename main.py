import os


def menu():
    print("\n=== MENU ===")
    print("1. Question 1 : Charger les données contenues dans data1")
    print("2. Question 2 : Informations générales sur les données")
    print("3. Question 3 : Visualiser une image")
    print("4. Question 3b : Homogénéiser les images")
    print("5. Question 4 : Preprocessing des images")
    print("6. Question 5a : Séparer les sets d'entraînement et de validation")
    print("7. Question 5b : Explication de random_state")
    print("8. Modèle supervisé 1 : Entraîner un arbre de décision")
    print("9. Modèle supervisé 2 : Entraîner un SVM")
    print("10. Evaluation de l'entraînement : Accuracy")
    print("11. Evaluation de l'entraînement : Matrice de confusion")
    print("12. Evaluation de l'entraînement : Courbe ROC")
    print("13. Evaluation sur les données de test")
    print("14. Augmentation de données")
    print("15. Modèle non-supervisé : K-means")
    print("16. Exploration des datasets : CIFAR-10")
    print("17. Détection d’anomalies ~ Classification open-set")
    print("0. Quitter")


def execute_choice(choice):
    script_path = './src/'

    if choice == 1:
        os.system(f'python {script_path}data_loading.py')
    elif choice == 2:
        os.system(f'python {script_path}data_loading.py')
    elif choice == 3:
        os.system(f'python {script_path}visualise_image.py')
    elif choice == 4:
        os.system(f'python {script_path}homogenize_image.py')
    elif choice == 5:
        os.system(f'python {script_path}preprocessing_images.py')
    elif choice == 6:
        os.system(f'python {script_path}random_state.py')
    elif choice == 7:
        os.system(f'python {script_path}random_state.py')
    elif choice == 8:
        os.system(f'python {script_path}train_decision_tree.py')
    elif choice == 9:
        os.system(f'python {script_path}train_svm.py')
    elif choice == 10:
        os.system(f'python {script_path}evaluation_accuracy.py')
    elif choice == 11:
        os.system(f'python {script_path}evaluation_confusion_matrix.py')
    elif choice == 12:
        os.system(f'python {script_path}evaluation_precision_recall_roc.py')
    elif choice == 13:
        os.system(f'python {script_path}evaluation_test_data.py')
    elif choice == 14:
        os.system(f'python {script_path}data_augmentation.py')
    elif choice == 15:
        os.system(f'python {script_path}unsupervised_kmeans.py')
    elif choice == 16:
        os.system(f'python {script_path}dataset_exploration.py')
    elif choice == 17:
        os.system(f'python {script_path}open_set_classification.py')
    elif choice == 0:
        print("Au revoir !")
        exit()
    else:
        print("Choix non valide, veuillez réessayer.")


if __name__ == "__main__":
    while True:
        menu()
        try:
            choice = int(input("\nEntrez le numéro de la question à exécuter : "))
            execute_choice(choice)
        except ValueError:
            print("Veuillez entrer un numéro valide.")
