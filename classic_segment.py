import cv2
import numpy as np
import argparse
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from kymatio.torch import Scattering2D


def classic_segment(image_path):
    """

    Strat basé sur du watershed pour faire de la segmentatio, un peu dégueu, généré par chatgpt
        
    """

    # parameters
    output_dir = "seg_out"

    # check si l'image existe
    if not os.path.isfile(image_path):
        print(f"[!] Can't load image from {image_path}")
        return -1

    # check if output folder exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Charger l'image
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # Appliquer un seuillage pour obtenir le foreground
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphologie pour séparer les objets collés
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Détection du fond (background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Détection du premier plan (foreground)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Obtenir les marqueurs pour Watershed
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marquer les régions
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Important pour que le fond soit bien distingué
    markers[unknown == 255] = 0  # Les zones inconnues deviennent 0

    # Appliquer Watershed
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir pour affichage
    cv2.watershed(image, markers)
    image_color[markers == -1] = [255, 0, 0]  # Contours en rouge

    # Créer un masque final en noir et blanc
    mask = np.zeros_like(image_gray)
    mask[markers > 1] = 255  # Tous les pixels marqués comme objets

    # Sauvegarder le masque
    cv2.imwrite(f"{output_dir}/masque_watershed.png", mask)

    # Affichage côte à côte : Image originale vs Masque final
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Masque de segmentation")
    axes[1].axis("off")

    plt.show()


def improve_segmentation(image_path):
    """ """

    # parameters
    features_folder = "features"

    if not os.path.isdir(features_folder):
        os.mkdir(features_folder)

    #- SCAT PART---------------------------
    # preprocess image
    img = Image.open(image_path).convert('L')

    # Redimensionner l'image pour correspondre à la taille attendue par la scattering transform.
    # Ici, nous choisissons une taille de 128x128 pixels.
    img = img.resize((512, 512))

    # Conversion en tenseur PyTorch
    to_tensor = transforms.ToTensor()  # convertit l'image en tenseur de forme (1, H, W) pour une image en niveaux de gris
    img_tensor = to_tensor(img)  # forme : (1, 128, 128)

    # Ajouter une dimension batch (la scattering attend une forme (batch, channels, H, W))
    img_tensor = img_tensor.unsqueeze(0)  # forme finale : (1, 1, 128, 128)

    # 2. Application de la scattering transform
    # Pour une image de taille 128x128, on définit shape=(128,128) et choisit un nombre d'échelles, par exemple J=2.
    scattering = Scattering2D(J=1, shape=(512, 512))

    # Appliquer la scattering transform sur l'image
    features = scattering(img_tensor)  # forme : (1, C, H_out, W_out)
    # Pour afficher, on récupère les features du premier (et unique) élément du batch
    features = features[0].cpu().detach().numpy()  # forme : (C, H_out, W_out)

    # save features
    print("Dimensions des features extraites :", features.shape)
    cmpt = 0
    for f in features[0]:
        plt.imsave(f"{features_folder}/features_scattering_{cmpt}.png", f, cmap="viridis")

        #- TODO SEGMENTATION ---------------------------
        # Appliquer Watershed

        # image_color = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)  # Convertir pour affichage
        # cv2.watershed(f, markers)
        # image_color[markers == -1] = [255, 0, 0]  # Contours en rouge

        # # Créer un masque final en noir et blanc
        # mask = np.zeros_like(image_gray)
        # mask[markers > 1] = 255  # Tous les pixels marqués comme objets
        
        cmpt +=1




if __name__ == "__main__":

    # Argument parser pour récupérer le chemin de l'image
    parser = argparse.ArgumentParser(description="Segmentation d'image avec Watershed")
    parser.add_argument("image_path", type=str, help="Chemin de l'image à segmenter")
    args = parser.parse_args()

    image_path = args.image_path

    # run classic segmentation
    # classic_segment(image_path)
    improve_segmentation(image_path)
    
