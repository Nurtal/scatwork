import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from kymatio.torch import Scattering2D

# 1. Chargement et prétraitement de l'image
# Charger l'image et la convertir en niveaux de gris
image_path = "/home/n765/Images/chu.png"
img = Image.open(image_path).convert('L')

# Redimensionner l'image pour correspondre à la taille attendue par la scattering transform.
# Ici, nous choisissons une taille de 128x128 pixels.
img = img.resize((128, 128))

# Conversion en tenseur PyTorch
to_tensor = transforms.ToTensor()  # convertit l'image en tenseur de forme (1, H, W) pour une image en niveaux de gris
img_tensor = to_tensor(img)  # forme : (1, 128, 128)

# Ajouter une dimension batch (la scattering attend une forme (batch, channels, H, W))
img_tensor = img_tensor.unsqueeze(0)  # forme finale : (1, 1, 128, 128)

# 2. Application de la scattering transform
# Pour une image de taille 128x128, on définit shape=(128,128) et choisit un nombre d'échelles, par exemple J=2.
scattering = Scattering2D(J=2, shape=(128, 128))

# Appliquer la scattering transform sur l'image
features = scattering(img_tensor)  # forme : (1, C, H_out, W_out)
# Pour afficher, on récupère les features du premier (et unique) élément du batch
features = features[0].cpu().detach().numpy()  # forme : (C, H_out, W_out)

print("Dimensions des features extraites :", features.shape)

for x in features[0]:

    plt.imshow(x, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # optional: adds a colorbar to show the scale
    plt.title("2D Array Visualization")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    plt.close()


# 3. Affichage des features
# Puisque le scattering transform produit plusieurs canaux, nous pouvons afficher chaque canal comme une image.
# On va organiser l'affichage dans une grille.

# num_channels = features.shape[0]
# # Déterminer la grille d'affichage : par exemple, environ sqrt(n_channels) par côté.
# grid_cols = int(np.ceil(np.sqrt(num_channels)))
# grid_rows = int(np.ceil(num_channels / grid_cols))

# fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 15))
# axes = np.array(axes).flatten()  # mettre sous forme de liste pour itérer facilement

# for i in range(num_channels):
#     ax = axes[i]
#     im = ax.imshow(features[i], cmap='viridis')
#     ax.set_title(f"Canal {i}")
#     ax.axis('off')
#     fig.colorbar(im, ax=ax)

# # Cacher les sous-graphes inutilisés s'il en reste
# for j in range(i+1, len(axes)):
#     axes[j].axis('off')

# plt.tight_layout()
# plt.show()

