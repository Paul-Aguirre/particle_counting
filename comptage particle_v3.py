import numpy as np
from nd2reader import ND2Reader
from skimage import filters, measure, morphology, draw
import matplotlib.pyplot as plt

# Chemin d'accès au fichier ND2
nd2_file_path = (
    r"C:\Users\aguirrep\Documents\microscopie\fluo_tests\comptage\1.5pc-.nd2"
)

# Lire l'image ND2 avec nd2reader
with ND2Reader(nd2_file_path) as images:
    # Obtenir le nombre de frames et la taille de chaque frame
    num_frames = len(images)
    frame_shape = images.shape
    print(f"Nombre de frames: {num_frames}, Taille d'une frame: {frame_shape}")

    # Empiler les tranches en une seule image 3D
    image_stack = np.array([frame for frame in images])

# Vérifier la forme de l'image 3D
print(
    "Shape of the image stack:", image_stack.shape
)  # Cela devrait afficher (num_frames, H, W)

print(
    f"For image 50, max = {np.max(image_stack[49,:,:])},"
    f"min = {np.min(image_stack[49,:,:])},"
    f"mean = {np.mean(image_stack[49,:,:])}"
)  # pour afficher quelques données sur les images traitées

# Appliquer un seuil global
# thresh = 2400 # pour faire le seuillage manuel
thresh = filters.threshold_otsu(image_stack)
print(f"Threshold value: {thresh}")
binary = image_stack > thresh

# Diamètre de la particule en micromètres (µm)
particle_diameter = 1  # Remplacer par le diamètre réel de vos particules

# Taille de l'image en pixels
image_size = image_stack.shape

# Résolution spatiale en micromètres par pixel (µm/px)
resolution_xy = 0.33  # µm/px pour X et Y
resolution_z = 1  # µm entre chaque tranche pour Z

# Conversion du diamètre de la particule en pixels
particle_diameter_xy = particle_diameter / resolution_xy
particle_diameter_z = particle_diameter / resolution_z

# Définir le structuring element basé sur la taille des particules
structuring_element = morphology.ball(
    particle_diameter_xy / 2
)  # 3D structuring element

# Séparer les particules touchantes
binary_separated = morphology.binary_opening(binary, structuring_element)

# Labeliser les particules en 3D
labels = measure.label(binary_separated)

# Extraire les propriétés des particules
props = measure.regionprops(labels)

print(f"Props bboxes:")
for prop in props:
    print(prop.bbox)  # pour visualiser les props

# Compter les particules en 3D
num_particles = len(props)
print(f"Nombre de particules : {num_particles}")

# Calculer le volume total de l'échantillon en micromètres cubes (µm^3)
total_volume = (
    image_size[0]
    * resolution_z
    * image_size[1]
    * resolution_xy
    * image_size[2]
    * resolution_xy
)

# Calculer la concentration de particules par unité de volume (particules/µm^3)
particles_per_volume = num_particles / total_volume

# Afficher la concentration de particules par unité de volume
print(f"Concentration de particules : {particles_per_volume} particules/µm^3")

# Optionnel : Afficher une tranche de l'image originale et la version binaire
for i in range(0, image_stack.shape[0], 30):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Tranche de l'image originale
    ax[0].imshow(image_stack[i], cmap="gray")
    ax[0].set_title(f"Tranche {i} de l'image originale")

    # Tranche de l'image binaire avec annotations
    ax[1].imshow(binary_separated[i], cmap="gray")

    # Dessiner les contours des particules détectées
    for prop in props:
        if (
            prop.bbox[0] <= i <= prop.bbox[3]
        ):  # Vérifier si la tranche intersecte la particule
            minz, minr, minc, maxz, maxr, maxc = prop.bbox
            # car minz et maxz sont donnés en premier
            if minz <= i <= maxz:
                rr, cc = draw.rectangle_perimeter(
                    start=(minr, minc), end=(maxr, maxc), shape=image_stack[i].shape
                )
                ax[1].plot(cc, rr, color="red")

    ax[1].set_title(f"Tranche {i} de l'image binaire avec particules détectées")

    plt.show()

fig3D, ax3D = plt.subplots(subplot_kw={"projection": "3d"})
ax3D.voxels(binary.transpose((1, 2, 0)))  # pour mettre l'axe z verticalement
# matplotlib aime pas trop le fait que le z soit en premier indice
# c'est pour ça qui faut faire ça
ax3D.set_xlabel("x")
ax3D.set_ylabel("y")
ax3D.set_zlabel("z")
# plt.show()

bboxes3d = np.zeros(binary.shape, dtype=np.uint8)
for prop in props:
    minz, minr, minc, maxz, maxr, maxc = prop.bbox
    zz, rr, cc = draw.rectangle(
        start=(minz, minr, minc),
        end=(maxz, maxr, maxc),
        shape=binary.shape,
    )
    bboxes3d[zz, rr, cc] = 1

ax3D.voxels(bboxes3d.transpose((1, 2, 0)), alpha=0.5)
plt.show()
