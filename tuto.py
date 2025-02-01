import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kymatio.torch import Scattering2D
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Définition des transformations sur les images : conversion en tenseur
transform = transforms.Compose([
    transforms.ToTensor()
])

# Téléchargement des datasets MNIST (train et test)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Création des DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Dimensions des images MNIST
img_height, img_width = 28, 28
image_shape = (img_height, img_width)

# Initialisation de la scattering transform
# Ici, J définit le nombre d'échelles. Pour MNIST, J=2 est un bon compromis.
scattering = Scattering2D(J=2, shape=image_shape)

def compute_scattering_features(loader):
    """
    Pour chaque batch du loader, applique la scattering transform,
    aplatit le tenseur de features et retourne un tableau NumPy des features et des labels.
    """
    features_list = []
    labels_list = []
    
    # On désactive le calcul des gradients pour plus de rapidité
    with torch.no_grad():
        for images, labels in loader:
            # images : [batch_size, 1, 28, 28]
            # Appliquer la scattering transform
            scattering_features = scattering(images)
            # Aplatir les features par image : on conserve le batch_size en première dimension
            scattering_features = scattering_features.view(scattering_features.size(0), -1)
            
            # Convertir en numpy
            features_list.append(scattering_features.numpy())
            labels_list.append(labels.numpy())
    
    # Concaténer tous les batches
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels

# Calcul des features pour les ensembles d'entraînement et de test
print("Calcul des features scattering pour l'ensemble d'entraînement...")
train_features, train_labels = compute_scattering_features(train_loader)
print("Calcul des features scattering pour l'ensemble de test...")
test_features, test_labels = compute_scattering_features(test_loader)

# Affichage de la dimension des features extraites
print("Dimension des features scattering :", train_features.shape[1])

# Classification avec une régression logistique
print("Entraînement du classifieur...")
clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
clf.fit(train_features, train_labels)

# Prédictions sur l'ensemble de test et évaluation de la précision
predictions = clf.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f"Précision sur l'ensemble de test : {accuracy:.4f}")

