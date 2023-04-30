import os
import numpy as np
from matplotlib import pyplot as plt

# This file directory
curr_dir = os.path.dirname(os.path.realpath(__file__))
# Dataset directory
data_dir = curr_dir + "/datasets"
dataset_zip = np.load(
    data_dir + "/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    allow_pickle=True,
    encoding="latin1",
)

# Print dataset keys
print(f"Keys in the dataset: {list(dataset_zip.keys())}")
# Extracting some info from dataset
imgs = dataset_zip["imgs"]
latents_values = dataset_zip["latents_values"]
latents_classes = dataset_zip["latents_classes"]
metadata = dataset_zip["metadata"][()]
# print("Metadata: \n", metadata)

# Number of values per latents (latents_sizes = array([ 1,  3,  6, 40, 32, 32]))
latents_sizes = metadata["latents_sizes"]

# Creating all bases, for mapping from latents values to img index in the dateset.
# This reflects the number of values for each latent and how the dataset was generated.
# The dataset was procedurally generated so that the image at index 0 has latent values [0, 0, 0, 0, 0, 0].
# Then only the value of the last latent was changed, [0, 0, 0, 0, 0, 1], producing the img at index 1.
# The img at index 31 has latents [0, 0, 0, 0, 0, 31], at index 32 you have [0, 0, 0, 0, 1, 0], at index 33 you have [0, 0, 0, 0, 1, 1] etc..
# This way images with all possible combinations of latents were created.
# Latents [0, 0, 0, 0, 0, 1] means taking/producing the image in which the first 5 latents have their zeroth values and the last takes its first value.
latents_bases = np.concatenate(
    (
        latents_sizes[::-1].cumprod()[::-1][1:],
        np.ones(1, dtype=np.int64),
    )
)
# Further notes:
# - ::-1 reverses the last dim order so latents_sizes = array([32,32,40,6,3,1])
# - cumprod() returns the cumulative product of the array elements including intermediate results
# - cumprod()[::-1][1:] flip again the order and get rid of the last result because it is the same as the second last result
# - add back a one-dimensional dimension to the array by concatenation so latent_bases = np.array([737280, 245760, 40960, 1024, 32, 1])

# Function to sample latents values
def sample_latent(size=1):
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)

    return samples


# Functions to convert latents values to img index in the dataset
def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


# Helper function to show images
def show_images_grid(imgs_, num_images=25):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap="Greys_r", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    plt.show()


def show_density(imgs):
    _, ax = plt.subplots()
    ax.imshow(imgs.mean(axis=0), interpolation="nearest", cmap="Greys_r")
    ax.grid("off")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# Sample latents randomly
latents_sampled = sample_latent(size=5000)

# Select images
indices_sampled = latent_to_index(latents_sampled)
imgs_sampled = imgs[indices_sampled]

# Show images
show_images_grid(imgs_sampled)
