import numpy as np
import torch
from torchvision import transforms
from kymatio.torch import Scattering2D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2


"""
ChatGPT déraille pas mal sur la fin, faire gaffe déjà à importer Scattering2D depuis kymatio.torch

    
"""


# 🔹 1. Charger l'image en niveaux de gris
image_path = "/home/n765/Images/francs.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Vérifier la forme de l'image
print("Shape de l'image avant redimensionnement :", image.shape)

# Redimensionner à 256x256 si nécessaire
image_resized = cv2.resize(image, (256, 256))  # Redimensionner pour simplifier
print("Shape après redimensionnement :", image_resized.shape)

# Normaliser l'image
image_resized = image_resized.astype(np.float32) / 255.0

# Appliquer le Scattering Transform
J = 2  # Nombre d'échelles
shape = image_resized.shape  # Dimensions de l'image
scattering = Scattering2D(J=J, shape=shape)

# Convertir en tenseur PyTorch
image_tensor = torch.tensor(image_resized).unsqueeze(0).unsqueeze(0)  # (batch, channel, height, width)

# Appliquer Scattering Transform
features = scattering(image_tensor).detach().numpy()  # Convertir en numpy

# Vérifier la forme des features extraites
print("Shape des features extraites par Scattering Transform :", features.shape)

# Aplatir les features (les dimensions doivent être [nb_pixels, nb_features])
features_flat = features.flatten().reshape(-1, features.shape[1])  # Flatten les features pour chaque pixel

# Vérifier la nouvelle forme
print("Shape des features aplaties :", features_flat.shape)

# Effectuer la segmentation avec KMeans sur les features aplatis
kmeans_scattering = KMeans(n_clusters=2, random_state=0).fit(features_flat)

# Redimensionner les labels de segmentation pour avoir la même taille que l'image
seg_scattering = kmeans_scattering.labels_.reshape(image_resized.shape)  # Segmentation sur features scattering

# Afficher l'image originale et la segmentation
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image_resized, cmap='gray')
ax[0].set_title("Image Originale")
ax[1].imshow(seg_scattering, cmap='gray')
ax[1].set_title("Segmentation avec Scattering Transform")

for a in ax:
    a.axis('off')

plt.show()
