import cv2
import numpy as np

def load_mask(image_path):
    """
    Charge une image en niveaux de gris et la convertit en masque binaire.
    
    :param image_path: Chemin de l'image
    :return: Masque binaire (NumPy array)
    """
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask // 255  # Normalisation (0 ou 1)

def calculate_iou(mask_pred, mask_true):
    """
    Calcule l'IoU entre deux masques binaires.
    
    :param mask_pred: Masque prédit (NumPy array, 0 et 1)
    :param mask_true: Masque de vérité terrain (NumPy array, 0 et 1)
    :return: Valeur IoU (float)
    """
    intersection = np.logical_and(mask_pred, mask_true).sum()
    union = np.logical_or(mask_pred, mask_true).sum()
    
    return intersection / union if union != 0 else 1.0 if intersection == 0 else 0.0


if __name__ == "__main__":

    # Charger les masques depuis les images
    mask1 = load_mask("seg_out/masque_watershed.png")
    mask2 = load_mask("masks/square_mask.png")

    # revert mask
    mask2 = 1 - mask2

    # Calculer l'IoU
    iou_score = calculate_iou(mask1, mask2)
    print(f"IoU: {iou_score:.4f}")

