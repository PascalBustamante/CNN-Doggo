import numpy as np

def create_patches(image, patch_size):
    h, w, c = image.shape
    image = np.reshape(image, (h // patch_size, patch_size, w // patch_size, patch_size, c))
    image = np.transpose(image, (0, 2, 1, 3, 4))
    patches = np.reshape(image, (-1, patch_size, patch_size, c))
    return patches


