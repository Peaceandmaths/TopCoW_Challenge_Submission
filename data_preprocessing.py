""" Data preprocessing 
Outputs dataframe and saves it as npy file with merged data information 
( original + augmented)

"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import random
from scipy.stats import zscore
from scipy.spatial import KDTree
import open3d as o3d
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#from google.colab import files
from keras.utils import plot_model
from scipy.ndimage.morphology import binary_closing
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import keras.backend as K
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
import pandas as pd

""" uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
     """

# Preparation of labels and label colors 

from matplotlib.colors import ListedColormap, Normalize
# Updated the label_mapping dictionary to include label 14 with a default value
label_mapping = {
    0: 'Background',
    1: 'BA',
    2: 'R-PCA',
    3: 'L-PCA',
    4: 'R-ICA',
    5: 'R-MCA',
    6: 'L-ICA',
    7: 'L-MCA',
    8: 'R-Pcom',
    9: 'L-Pcom',
    10: 'Acom',
    11: 'R-ACA',
    12: 'L-ACA',
    13: 'Label-13',
    14: 'Label-14',
    15: '3rd-A2',
}

# Updated label_colors to account for the additional label
label_colors = [
    'black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow',
    'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'teal', 'lime', 'beige'
]

# Paths 

data_path = '/home/golubeka/TopCoW Challenge/TopCoW-Challenge/imagesTr'
mask_path = '/home/golubeka/TopCoW Challenge/TopCoW-Challenge/mul_labelsTr'

images = os.listdir(data_path) 
masks = os.listdir(mask_path)

nifti_files = [file for file in images if file.startswith('topcow_ct_whole_')]
nifti_masks = [file for file in masks if file.startswith('topcow_ct_whole_')]
# Sort the files to ensure consistent pairing of image and mask 
nifti_files.sort()
nifti_masks.sort()




#################################################################
# Getting voxel label and data # NEW
################################################################

def create_point_cloud(image_data):
    print("Entered create_point_cloud")
    # Create an empty list to store the voxel data and labels
    voxel_data = []
    voxel_label = []

    # Get the dimensions of the image volume
    num_slices_x, num_slices_y, num_slices_z = image_data.GetSize()

    # Iterate through all voxel coordinates
    for x in range(num_slices_x):
        for y in range(num_slices_y):
            for z in range(num_slices_z):
                if x < image_data.shape[0]and y < image_data.shape[1] and z < image_data.shape[2]:
                    x_coord = x
                    y_coord = y
                    z_coord = z
                    pixel_intensity = image_data[x, y, z]
                    #annotation_label = mask_data[x, y, z]
                    voxel_data.append([x_coord, y_coord, z_coord, pixel_intensity])
                    #voxel_label.append(annotation_label)

    # Convert voxel_label to integer
    #voxel_label = np.array(voxel_label, dtype=np.int32)
    voxel_data_array = np.array(voxel_data)

    # Reshape voxel_label to have two dimensions
    #voxel_label_reshaped = voxel_label[:, np.newaxis]
    #voxel_label_data = voxel_label_reshaped

    # Merge data and one-hot labels
    point_cloud_data = np.hstack((voxel_data_array))

    return point_cloud_data

import numpy as np

def check_annotation_presence(mask_data):
    # Get the unique values in the mask_data array
    unique_labels = np.unique(mask_data)

    # Print the unique annotation labels
    print("Unique Annotation Labels:", unique_labels)

    # Convert the unique_labels to integers (since they might be float values)
    unique_labels = np.array(unique_labels, dtype=int)

    # Filter out the background label (0)
    unique_labels = unique_labels[unique_labels != 0]

    # Print the unique annotation labels excluding the background
    print("Unique Annotation Labels (excluding Background):", unique_labels)

    annotations_of_interest = [1, 2, 3, 4,5,6,7,8, 9, 10, 11, 12,13,14,15] 
    # Check if specific annotation labels are present
    for annotation_label in annotations_of_interest:
        if annotation_label in unique_labels:
            print(f"Annotation Label {annotation_label} is present.")
        else:
            print(f"Annotation Label {annotation_label} is not present.")


################################
# Preprocessing functions ####
################################

def preprocess_point_cloud(point_cloud_data):
    """
    Preprocesses a point cloud dataset.

    Parameters:
        point_cloud_data (numpy.ndarray): Input point cloud data in the form [x, y, z, intensity, label].

    Returns:
        numpy.ndarray: Preprocessed point cloud data after applying various processing steps.
    """
# 1. Removing points with missing or invalid co-ordinates
    """
    Remove points with missing or invalid coordinates from a point cloud.

    Args:
        point_cloud (numpy.ndarray): Input point cloud with shape (N, 5), where N is the number of points and
                                    each row contains [x, y, z, intensity, label].

    Returns:
        numpy.ndarray: Cleaned point cloud with invalid points removed.

    Notes:
        This function removes points from the input point cloud where the intensity value is -999.95960192,
        which is assumed to represent missing or invalid data.
    """

    def remove_invalid_points(point_cloud_data):
        valid_mask = point_cloud_data[:, 3] != -999.95960192
        cleaned_point_cloud = point_cloud_data[valid_mask]
        print("Shape after removing invalid points:", cleaned_point_cloud.shape)
        return cleaned_point_cloud

# 2. Apply outlier detection (using a simple z-score threshold as an example)
    """
        Apply outlier detection to a point cloud using a z-score threshold.

        Args:
            point_cloud (numpy.ndarray): Input point cloud with shape (N, 5), where N is the number of points and
                                        each row contains [x, y, z, intensity, label].
            z_threshold (float, optional): Z-score threshold for outlier detection.

        Returns:
            numpy.ndarray: Point cloud with outliers removed.

    """

    def remove_outliers(cleaned_point_cloud, z_threshold=3):
        z_scores = zscore(cleaned_point_cloud[:, :3], axis=0)  # Calculate z-scores along axis 0
        valid_mask = np.all(np.abs(z_scores) < z_threshold, axis=1)
        filtered_point_cloud = cleaned_point_cloud[valid_mask]
        print("Shape after removing outliers points:", filtered_point_cloud.shape)
        return filtered_point_cloud

# 3. Normalize spatial coordinates to a specific range (e.g., [-1, 1])
    """
        Normalize spatial coordinates of a point cloud to a specific range.

        Args:
            point_cloud (numpy.ndarray): Input point cloud with shape (N, 5), where N is the number of points and
                                        each row contains [x, y, z, intensity, label].

        Returns:
            numpy.ndarray: Point cloud with normalized coordinates.

    """
    def normalize_coordinates(point_cloud):
        min_coords = np.min(point_cloud[:, :3], axis=0)
        max_coords = np.max(point_cloud[:, :3], axis=0)
        center = (max_coords + min_coords) / 2
        scale = np.max(max_coords - min_coords) / 2
        normalized_coords = (point_cloud[:, :3] - center) / scale
        print("Shape after normalizing coordinates:", normalized_coords.shape)
        return normalized_coords


# 4. Filter background labels from true labels

"""
    Filters out background points (label 0) from the point cloud data.

    Parameters:
        point_cloud (numpy.ndarray): Point cloud data in the form [x, y, z, intensity, label].

    Returns:
        numpy.ndarray: Point cloud data with background points removed.
"""

def filter_background_points(point_cloud):
    non_background_mask = point_cloud[:, -1] != 0
    true_points = point_cloud[non_background_mask]
    print("True_points :", true_points[:10])
    return true_points



# 4.Define the function to normalize labels and ignore background (label 0)
"""
    Normalizes labels by subtracting 1 to ignore the background label (label 0).

    Parameters:
        labels (numpy.ndarray): Array of labels.

    Returns:
        numpy.ndarray: Normalized labels.
"""

def normalize_labels_ignore_background(labels):
    normalized_labels = np.copy(labels)-1
    normalized_labels = normalized_labels + 1
    return normalized_labels

# 5.To normalise and extract their labels
"""
    Normalizes labels and applies logarithmic transformation to intensity values of the point cloud data.

    Parameters:
        true_points (numpy.ndarray): Point cloud data without background points.

    Returns:
        numpy.ndarray: Preprocessed point cloud data.
"""

def normalize_intensity(true_points):
    # Create a copy of the true_points array
    preprocessed_points = np.copy(true_points)

    # Normalize labels and extract normalized labels
    #normalized_labels = normalize_labels_ignore_background(preprocessed_points[:, -1])
    #preprocessed_points[:, -1] = normalized_labels

    # Checking for negative values
    has_negative_values = np.any(preprocessed_points[:, 3] < 0)

    if has_negative_values:
        print("Negative values are present in the preprocessed_point_cloud.")
    else:
        print("No negative values are present in the preprocessed_point_cloud.")

    # Apply logarithmic transformation to intensity values
    min_intensity = abs(np.min(preprocessed_points[:, 3]))
    shifted_intensities = preprocessed_points[:, 3] - min_intensity
    log_transformed_intensities = np.log1p(shifted_intensities)

    # Update the intensity values in the point cloud with the transformed values
    preprocessed_points[:, 3] = log_transformed_intensities

    return preprocessed_points


################################################################
# Loading preproocessed data set for augmentations #
################################################################


def load_custom_data_cls(preprocessed_point_cloud):
    all_data = preprocessed_point_cloud[:, :4]  # Extract x, y, z, intensity
    #all_label = preprocessed_point_cloud[:, 4]   # Extract labels
    return all_data



""" Checking one-hot encoded voxelized labels shape
for _ in range(5):
    i = random.randint(0, len(preprocessed_point_cloud) - 1)
    print(f"preprocessed_point_cloud[{i}].shape:", preprocessed_point_cloud[i].shape)
    print(f"voxel_label_onehot[{i}].shape:", voxel_label_onehot[i].shape)

    # Displaying the original label and its one-hot encoded form.
    print(
        f"Original voxel_label[{i}]:", voxel_label[i],
        f"\tOne-hot voxel_label_onehot[{i}]:", voxel_label_onehot[i]
    ) """


#########################################################
# Augmentations functions
#########################################################

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud[:, :3], xyz1), xyz2).astype('float32')
    return np.hstack((translated_pointcloud, pointcloud[:, 3:]))

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    jitter = np.clip(sigma * np.random.randn(N, C - 2), -clip, clip)
    jittered_pointcloud = pointcloud.copy()
    jittered_pointcloud[:, :3] += jitter
    return jittered_pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_points = pointcloud[:, :2].dot(rotation_matrix)
    rotated_pointcloud = np.hstack((rotated_points, pointcloud[:, 2:]))
    return rotated_pointcloud

def augment_point_cloud(points, num_augmentations):
    augmented_point_clouds = []

    for _ in range(num_augmentations):
        translated_points = translate_pointcloud(points)
        jittered_points = jitter_pointcloud(translated_points)
        rotated_points = rotate_pointcloud(jittered_points)

        augmented_point_clouds.extend([translated_points, jittered_points, rotated_points])
        #augmented_labels.extend([labels] * 3)  # Use the same labels for each augmented point cloud

    return np.array(augmented_point_clouds)



#########################################################
# Augmentations on preprocessed point cloud whole dataset
#########################################################


""" Checking augmented data shapes 
print(augmented_data_point_clouds[:3])
print(preprocessed_point_cloud[:3])

print("Original point cloud shape:", preprocessed_point_cloud.shape)
print("Original labels shape:", all_labels.shape)
print("Shape of Augmented point clouds with 1 augmentation per datapoint:", augmented_data_point_clouds.shape)
augmented_shape = (len(augmented_point_clouds),) + augmented_point_clouds[0].shape[1:]
print("Augmented point clouds shape:", augmented_shape)
print("Augmented labels shape:", augmented_data_labels.shape)

 """
########################################################################################
# Augmentations on batches of preprocessed point cloud for visualisation purposes      #
########################################################################################

def translate_pointcloud(point_cloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(point_cloud[:, :3], xyz1), xyz2).astype('float32')
    return np.hstack((translated_pointcloud, point_cloud[:, 3:]))

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    jitter = np.clip(sigma * np.random.randn(N, C), -clip, clip)
    jittered_pointcloud = pointcloud + jitter
    return jittered_pointcloud

def rotate_pointcloud(point_cloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_points = point_cloud[:, :2].dot(rotation_matrix)
    rotated_pointcloud = np.hstack((rotated_points, point_cloud[:, 2:]))
    return rotated_pointcloud

def augment_batch(batch_data, batch_labels, num_augmentations):
    augmented_batch_points = []
    augmented_batch_labels = []

    for _ in range(num_augmentations):
        translated_points = translate_pointcloud(batch_data)
        jittered_points = jitter_pointcloud(translated_points)
        rotated_points = rotate_pointcloud(jittered_points)

        augmented_batch_points.extend([translated_points, jittered_points, rotated_points])
        augmented_batch_labels.extend([batch_labels] * 3)  # Use the same labels for each augmented point cloud

    return np.array(augmented_batch_points), np.array(augmented_batch_labels)


# Augmentation on batches for visualization 

""" batch_size = 32

# Generate batches of random indices for data and labels
num_samples = len(all_data)
indices = np.arange(num_samples)
np.random.shuffle(indices) """

""" for start_idx in range(0, num_samples - batch_size + 1, batch_size):
    excerpt = indices[start_idx:start_idx + batch_size]
    batch_data = all_data[excerpt]
    batch_labels = all_labels[excerpt]

    augmented_batch_points, augmented_batch_labels = augment_batch(batch_data, batch_labels, num_augmentations)

    # Now you have augmented batches of data and corresponding labels
    # Use these batches for training or any other purpose
    #print("Augmented Batch Points Shape:", augmented_batch_points.shape)
    #print("Augmented Batch Labels Shape:", augmented_batch_labels.shape)

 """
""" #######################################################
# Visualization of original and augmented point clouds#
#######################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


###################################
#Visualisation before augmnetation#
###################################
original_point_cloud = preprocessed_point_cloud


# Separate the points by label for visualization
points_by_label = {}
for point in preprocessed_point_cloud:
    label = int(point[-1])
    if label not in points_by_label:
        points_by_label[label] = []
    points_by_label[label].append(point[:-1])  # Exclude the label for plotting

# Create a 3D scatter plot for both the original and augmented point clouds
fig = plt.figure(figsize=(15, 6))

# Original point cloud subplot
ax1 = fig.add_subplot(121, projection='3d')
for label, points in points_by_label.items():
    points = np.array(points)
    ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=label_colors[label] if label < len(label_colors) else 'gray',
        label=f'Label {label_mapping[label]}',
        marker='o', s=20, alpha=0.7
    )

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Original Point Cloud')
ax1.legend()


# Manually adjust layout
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

# Show the plot
plt.show() """


"""  #########################################
#Visualisation with augmnetation overlay#
#########################################
import numpy as np

# Generate sample data (replace this with your actual data)
original_data = np.concatenate((preprocessed_point_cloud[:, :4], preprocessed_point_cloud[:, -1].reshape(-1, 1)), axis=1)  # Shape: (num_points, num_features)
augmented_data = np.random.rand(3, 32, 4)  # Shape: (num_augmentations, num_points, num_features)

# Visualize the original point cloud
fig = plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
ax = fig.add_subplot(111, projection='3d')
for i in range(original_data.shape[0]):
    point = original_data[i]
    label = int(point[-1])
    color = label_colors[label]
    ax.scatter(point[0], point[1], point[2], c=color, s=10, marker='o')

# Visualize the augmented point clouds
for i in range(augmented_batch_points.shape[0]):
    for j in range(augmented_batch_points.shape[1]):
        point = augmented_batch_points[i, j]
        label = int(augmented_batch_labels[i, j])
        color = label_colors[label]
        ax.scatter(point[0], point[1], point[2], c=color, s=10, marker='x')

# Adjust axis limits for better visibility
ax.set_xlim(min(original_data[:, 0]), max(original_data[:, 0]))
ax.set_ylim(min(original_data[:, 1]), max(original_data[:, 1]))
ax.set_zlim(min(original_data[:, 2]), max(original_data[:, 2]))

# Manually adjust layout
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Overlay of Original and Augmented Point Clouds')

# Show the plot
plt.show() """

################################################################
#    Merging original and augmented point clouds               #
################################################################



def compute_intensity_variation(points, intensities, k=5):
    tree = KDTree(points)
    intensity_variation_map = {}
    for point, intensity in zip(points, intensities):
        _, idxs = tree.query(point, k=k)  # Query k nearest neighbors
        avg_intensity = np.mean(intensities[idxs])
        intensity_variation_map[tuple(point)] = abs(intensity - avg_intensity)
    return intensity_variation_map

def compute_normals(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals

def point_cloud_to_normals_with_intensity(xyz, all_intensity):
    # Compute normals based purely on XYZ coordinates
    normals = compute_normals(xyz)
    # Append intensity as an additional feature to normals
    enhanced_normals = np.concatenate([normals, all_intensity[:, np.newaxis]], axis=-1)
    return enhanced_normals

def load_custom_model_train(merged_data):
    xyz = merged_data[:, :3]  # Extract x, y, z
    all_intensity = merged_data[:, 3]
    all_label = merged_data[:, 4].astype(np.int32)  # Extract labels

    # Compute interaction terms
    interaction_terms = np.multiply(xyz, all_intensity[:, np.newaxis])  # Ensure intensity has the right shape for element-wise multiplication

    num_classes = np.max(all_label) + 1  # assuming labels start from 0
    label_onehot = keras.utils.to_categorical(all_label, num_classes=num_classes)

    #Enhanced normals
    enhanced_normals = point_cloud_to_normals_with_intensity(xyz, all_intensity)

    # Compute interaction terms
    interaction_terms = np.multiply(xyz, all_intensity[:, np.newaxis])  # Ensure intensity has the right shape for element-wise multiplication

    #Compute Enhanced intensity
    intensity_variation_map = compute_intensity_variation(xyz, all_intensity)
    variations = [intensity_variation_map[tuple(p)] for p in xyz]

    # Concatenate all information
    all_data = np.concatenate([xyz, all_intensity[:, np.newaxis], enhanced_normals, np.array(variations)[:, np.newaxis], interaction_terms, label_onehot], axis=-1)

    return all_data


""" #Bounding box and Point Spread

# Compute bounding box
min_vals = all_data_merged[:, :3].min(axis=0)
max_vals = all_data_merged[:, :3].max(axis=0)

print(f"Bounding Box Min values: {min_vals}")
print(f"Bounding Box Max values: {max_vals}")

# Compute range for each dimension
ranges = max_vals - min_vals
print(f"Ranges (x, y, z): {ranges}") """

from scipy.spatial import KDTree



def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """
    Convolutional block with batch normalization and ReLU activation.

    Parameters:
    - x (tf.Tensor): Input tensor.
    - filters (int): Number of filters for the convolutional layer.
    - name (str): Prefix for the layer names.

    Returns:
    - tf.Tensor: Output tensor after applying convolution, batch normalization, and ReLU activation.
    """

    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """
    Multi-layer perceptron (MLP) block with batch normalization and ReLU activation.

    Parameters:
    - x (tf.Tensor): Input tensor.
    - filters (int): Number of units for the dense layer.
    - name (str): Prefix for the layer names.

    Returns:
    - tf.Tensor: Output tensor after applying dense layer, batch normalization, and ReLU activation.
    """
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """
    Regularizer to enforce orthogonality in transformation matrices.

    This regularizer ensures that the learned transformation matrix is close to an orthogonal matrix.
    """
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config

def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Constructs a transformation network to learn an affine transformation matrix.

    Parameters:
    - inputs (tf.Tensor): Input tensor.
    - num_features (int): Number of features for the output transformation matrix.
    - name (str): Prefix for the layer names.

    Returns:
    - tf.Tensor: Output tensor representing the learned transformation matrix.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)

def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Applies the learned transformation matrix to the input tensor.

    Parameters:
    - inputs (tf.Tensor): Input tensor.
    - num_features (int): Number of features for the transformation matrix.
    - name (str): Prefix for the layer names.

    Returns:
    - tf.Tensor: Output tensor after applying the transformation.
    """
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(transformed_features)
    return tf.matmul(inputs, transformed_features)



class MultiHeadAttentionBlock(keras.layers.Layer):
    """
    Multi-head attention block for point cloud data.

    This layer applies multi-head self-attention to the input tensor.
    """
    def __init__(self, num_heads, num_features, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_features = num_features
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.num_features // self.num_heads,
            value_dim=self.num_features // self.num_heads,
            dropout=self.dropout_rate,
        )
        super(MultiHeadAttentionBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        query = inputs  # Use 'inputs' as the 'query'
        key = inputs  # Use 'inputs' as the 'key'
        value = inputs  # Use 'inputs' as the 'value'
        attention_output = self.attention(
            query=query,
            key=key,
            value=value,
            training=training,
        )
        return attention_output

    def compute_output_shape(self, input_shape):
        return input_shape





def average_point_spacing(points, k=2):
    """Computes the average point spacing in the point cloud.

    Parameters:
    - points: The input point cloud.
    - k: Number of neighbors to consider (default is 2, which means the closest point).

    Returns:
    - The average spacing of the points.
    """
    tree = KDTree(points)
    # Query the KDTree to get the distance to the nearest point
    # Note: k=2 will give us the distance to the first point (itself) and the second point (the nearest point).
    dists, _ = tree.query(points, k=k)
    # We take all distances to the second closest point (ignoring distance to the point itself)
    avg_distance = np.mean(dists[:, 1])
    return avg_distance


def dense_voxelization(points, voxel_size):
    # Convert points to voxel indices
    voxel_indices = np.floor_divide(points, voxel_size).astype(np.int)

    # Create a dense voxel grid initialized to zero
    max_indices = np.max(voxel_indices, axis=0)
    voxel_grid = np.zeros((max_indices[0] + 1, max_indices[1] + 1, max_indices[2] + 1), dtype=np.int)

    # Fill the voxel grid based on point data
    for i, point in enumerate(points):
        voxel_grid[tuple(voxel_indices[i])] = 1  # or use the class label or feature

    return voxel_grid


def post_process(voxel_grid, structure=None):
    if structure is None:
        # Use a 3x3x3 cube as the default structure for 3D morphological operations
        structure = np.ones((3, 3, 3))
    return binary_closing(voxel_grid, structure=structure)


def dbscan_clustering_with_intensity(xyz, all_intensity, eps=0.03, min_points=10):
    enhanced_data = np.concatenate([xyz, all_intensity[:, np.newaxis]], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enhanced_data)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    return labels


def post_process(voxel_grid, structure=None):
    if structure is None:
        # Use a 3x3x3 cube as the default structure for 3D morphological operations
        structure = np.ones((3, 3, 3))
    return binary_closing(voxel_grid, structure=structure)

class PostSegmentationProcessingLayer(layers.Layer):
    def __init__(self, voxel_size, **kwargs):
        super(PostSegmentationProcessingLayer, self).__init__(**kwargs)
        self.voxel_size = voxel_size

    def call(self, inputs):
        # 1. DBSCAN clustering within segments
        clustered_labels = tf.numpy_function(self.apply_dbscan, [inputs[:, :3]], tf.float32)
        # 2. Smoothing - Gaussian filter
        smoothed_segmentation = tf.numpy_function(self.apply_gaussian, [inputs], tf.float32)
        # 3. Voxelization and post-processing
        voxelized_output = tf.numpy_function(self.apply_voxelization_and_post_process, [smoothed_segmentation], tf.float32)
        return voxelized_output

    def apply_dbscan(self, x):
        # Convert tensor to numpy
        x_np = x.numpy()
        return np.apply_along_axis(dbscan_clustering_with_intensity, 1, x_np)

    def apply_gaussian(self, x):
        x_np = x.numpy()
        return gaussian_filter(x_np, sigma=0.5)

    def apply_voxelization_and_post_process(self, x):
        voxel_grid = dense_voxelization(x, self.voxel_size)
        return post_process(voxel_grid)

    def compute_output_shape(self, input_shape):
        return input_shape





#def debug_shapes(model, input_tensor):
    #intermediate_model = keras.Model(inputs=model.input, outputs=[layer.output for layer in model.layers])
    #intermediate_outputs = intermediate_model.predict(input_tensor)

    #for layer, output in zip(model.layers, intermediate_outputs):
        #print(f"Layer {layer.name}: {output.shape}")

# Use the function
#batch_input_sample = np.random.rand(1, 2048, 13)  # Replace with a sample batch_input tensor
#debug_shapes(model_with_attention, batch_input_sample)


###################################################################################
###################################################################################

def average_point_spacing(points, k=2):
    """Computes the average point spacing in the point cloud.

    Parameters:
    - points: The input point cloud.
    - k: Number of neighbors to consider (default is 2, which means the closest point).

    Returns:
    - The average spacing of the points.
    """
    tree = KDTree(points)
    # Query the KDTree to get the distance to the nearest point
    # Note: k=2 will give us the distance to the first point (itself) and the second point (the nearest point).
    dists, _ = tree.query(points, k=k)
    # We take all distances to the second closest point (ignoring distance to the point itself)
    avg_distance = np.mean(dists[:, 1])
    return avg_distance
""" 
# Get the point data (assuming it's the first 3 columns of your data matrix)
points = all_data_merged[:, :3]
avg_spacing = average_point_spacing(points)

voxel_size = avg_spacing/10
print(voxel_size) """

#####################################################################################
#####################################################################################



# Extracting the segmented coordinates and intensities for each class
def extract_segmented_data(segmented_output, input_points):
    segmented_classes = np.argmax(segmented_output, axis=-1)  # Shape: (batch_size, 10240)

    segmented_coords_dict = {}
    segmented_intensities_dict = {}

    for class_id in range(13):  # 13 classes
        class_mask = segmented_classes == class_id

        # Extract coordinates and intensities for points that belong to the current class
        segmented_coords = []
        segmented_intensities = []

        for batch_idx in range(class_mask.shape[0]):
            coords_batch = input_points[batch_idx, class_mask[batch_idx], :3]  # x, y, z for this batch
            intensities_batch = input_points[batch_idx, class_mask[batch_idx], 3]  # Intensities for this batch

            segmented_coords.append(coords_batch)
            segmented_intensities.append(intensities_batch)

        segmented_coords_dict[class_id] = segmented_coords
        segmented_intensities_dict[class_id] = segmented_intensities

    return segmented_coords_dict, segmented_intensities_dict

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage, fcluster

def hierarchical_clustering(points, max_d=1.0, t=None, criterion='distance'):
    """
    points: the segmented point data
    max_d: maximum cophenetic distance between two clusters. If not None, it overrides t.
    t: The threshold to apply when forming flat clusters.
    criterion: The criterion to use in forming flat clusters.

    Returns cluster labels for each point.
    """
    Z = linkage(points, method='ward')
    if max_d:
        return fcluster(Z, t=max_d, criterion='distance')
    return fcluster(Z, t=t, criterion=criterion)

def dense_voxelization_with_clustering(points, cluster_labels, voxel_size):
    voxel_indices = np.floor_divide(points, voxel_size).astype(int)
    max_indices = np.max(voxel_indices, axis=0)

    # Initialize voxel grid to -1 indicating no assignment yet
    voxel_grid = np.full((max_indices[0] + 1, max_indices[1] + 1, max_indices[2] + 1), -1, dtype=int)

    # Use a dictionary to store votes for each voxel
    vote_dict = {}

    for i, point in enumerate(points):
        index = tuple(voxel_indices[i])
        if index not in vote_dict:
            vote_dict[index] = {}
        vote_dict[index][cluster_labels[i]] = vote_dict[index].get(cluster_labels[i], 0) + 1

    # Now assign the voxel values based on votes
    for index, votes in vote_dict.items():
        max_vote_class = max(votes, key=votes.get)
        voxel_grid[index] = max_vote_class

    return voxel_grid

# Refinement function
def refine_segmentation(segmented_voxels, structure=None):
    return binary_closing(segmented_voxels, structure=structure)

# Main Workflow
def process_point_cloud(segmented_output, input_points, voxel_size):
    # Extract
    coords_dict, intensities_dict = extract_segmented_data(segmented_output, input_points)

    refined_voxel_grids = {}
    labels_for_voxels = {}  # New dictionary to store label info

    for class_id in coords_dict:
        # Use hierarchical clustering instead of the undefined cluster_points function
        clustered_labels = hierarchical_clustering(np.vstack(coords_dict[class_id]))

        # Use dense_voxelization_with_clustering instead of dense_voxelization
        voxel_grid = dense_voxelization_with_clustering(np.vstack(coords_dict[class_id]), clustered_labels, voxel_size)
        refined_voxel_grid = refine_segmentation(voxel_grid)

        refined_voxel_grids[class_id] = refined_voxel_grid
        labels_for_voxels[class_id] = class_id  # Attach label info

    return refined_voxel_grids, labels_for_voxels

from scipy.spatial import KDTree
#pip install open3d
import open3d as o3d

def compute_intensity_variation(points, intensities, k=5):
    tree = KDTree(points)
    intensity_variation_map = {}
    for point, intensity in zip(points, intensities):
        _, idxs = tree.query(point, k=k)  # Query k nearest neighbors
        avg_intensity = np.mean(intensities[idxs])
        intensity_variation_map[tuple(point)] = abs(intensity - avg_intensity)
    return intensity_variation_map

def compute_normals(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals

def point_cloud_to_normals_with_intensity(xyz, all_intensity):
    # Compute normals based purely on XYZ coordinates
    normals = compute_normals(xyz)
    # Append intensity as an additional feature to normals
    enhanced_normals = np.concatenate([normals, all_intensity[:, np.newaxis]], axis=-1)
    return enhanced_normals

def load_custom_model_train(merged_data):
    xyz = merged_data[:, :3]  # Extract x, y, z
    all_intensity = merged_data[:, 3]
    all_label = merged_data[:, 4].astype(np.int32)  # Extract labels

    num_classes = np.max(all_label) + 1  # assuming labels start from 0
    label_onehot = keras.utils.to_categorical(all_label, num_classes=num_classes)

    #Enhanced normals
    enhanced_normals = point_cloud_to_normals_with_intensity(xyz, all_intensity)

    # Compute interaction terms
    interaction_terms = np.multiply(xyz, all_intensity[:, np.newaxis])  # Ensure intensity has the right shape for element-wise multiplication

    #Cmpute Enhanced intensity
    intensity_variation_map = compute_intensity_variation(xyz, all_intensity)
    variations = [intensity_variation_map[tuple(p)] for p in xyz]

    # Concatenate all information
    all_data = np.concatenate([xyz, all_intensity[:, np.newaxis], enhanced_normals, np.array(variations)[:, np.newaxis], interaction_terms, label_onehot], axis=-1)

    return all_data



#Data processing function

import numpy as np
import pandas as pd
import nibabel as nib

def batch_data(data, batch_size=10240):
        num_points = data.shape[0]
        batches = []

        for i in range(0, num_points, batch_size):
            batch = data[i:i+batch_size]

            # If the batch is smaller than the desired batch size, reuse points from the previous batch
            if batch.shape[0] < batch_size:
                deficit = batch_size - batch.shape[0]
                batch = np.concatenate([data[i-deficit:i], batch], axis=0)

            batches.append(batch)
        return batches
    
    
def create_image_data_dict(image_data):
    # Create an empty list to store image data
    image_data_dict = []

    # Split the data into num_points and point_clouds
    num_points = image_data[0]
    point_clouds = image_data[1]

    # Split the point_clouds into input_features and point_labels
    input_features = point_clouds[:, :3]
    point_labels = point_clouds[:, 3:]

    # Create a dictionary to store the image data
    image_data_dict.append({
        "num_points": num_points,
        "input_features": input_features,
        "point_labels": point_labels
    })

    return image_data_dict # maybe add num_classes = len(point_labels[0])


def concatenate_xyz_and_labels(input_points_list, label_predictions):
    concatenated_list = []

    for input_point, pred in zip(input_points_list, label_predictions):
        # Extract x, y, z coordinates from the input_points
        xyz = input_point[0, :, :3]  # We index [0] to remove the redundant dimension

        # Reshape the prediction to match the xyz shape
        label = pred[0, :].reshape(-1, 1)  # Reshaping to make it a column vector

        # Concatenate xyz with the label
        concatenated = np.hstack([xyz, label])  # hstack because we're adding a column
        concatenated_list.append(concatenated)

    # Convert the list of arrays into a single numpy array
    pred_array = np.vstack(concatenated_list)

    # Save the concatenated_data as a .npy file
    np.save('pred.npy', pred_array)

    return pred_array


def data_preprocessing_function(image_file):
  print("Entered data_preprocessing_function")

  # Initialize list to store data
  merged_data_list = []

  # Read mha image 
  # Replace 'your_image.mha' with the actual path to your MHA image file
  # Load the MHA image using MedPy
  image_data, _ = medpyload(image_file)
  image_data.transpose((2, 1, 0)).astype(np.uint8)

  # Now 'image_data' is a NumPy array containing the image pixel data
  
  # Get the dimensions of the image volume
  # num_slices_x, num_slices_y, num_slices_z = image_data.shape

    # Updated the color_map creation to ensure it covers all labels in mask_data
  #color_map = ListedColormap(label_colors[:int(np.max(mask_data)) + 1])

  point_cloud_data = create_point_cloud(image_data)
    # print(point_cloud_data.shape)
    # print(point_cloud_data[1:10])
    # check_annotation_presence(mask_data)
  raw_point_cloud = point_cloud_data
  #filtered_point_cloud = filter_background_points(raw_point_cloud)
    #print(filtered_point_cloud.shape)
  preprocessed_point_cloud = normalize_intensity(raw_point_cloud)
  #labels_data = (preprocessed_point_cloud[::,4].astype(np.int32))
    #print(labels_data[1:5])
    # One-hot encoding of labels
  # num_classes = np.max(labels_data) + 1  # assuming labels start from 0
  # voxel_label_onehot = keras.utils.to_categorical(labels_data, num_classes=num_classes)
  all_data = load_custom_data_cls(preprocessed_point_cloud)

    # Augmentations on preprocessed point cloud whole dataset
    # Assuming you have your 5D point cloud data in the 'points' variable (shape: [11408, 5])
  num_augmentations = 3  # Number of augmentations per data point

  augmented_point_clouds = []

  for i in range(len(preprocessed_point_cloud)):
    sample_point_cloud = np.array([preprocessed_point_cloud[i]])  # Select one data point at a time
    augmented_data, augmented_label = augment_point_cloud(sample_point_cloud, num_augmentations)
    augmented_point_clouds.extend(augmented_data)
    #augmented_labels.extend(augmented_label)

  augmented_data_point_clouds = np.array(augmented_point_clouds)
  #augmented_data_labels = np.array(augmented_labels)

    # Reshape the augmented point clouds to remove the extra dimension
  augmented_data_point_clouds_reshaped = augmented_data_point_clouds.reshape((-1, 5))

    #print(augmented_data_point_clouds_reshaped[1:3])

    # Merge the preprocessed and augmented point cloud data
  merged_data = np.vstack((preprocessed_point_cloud, augmented_data_point_clouds_reshaped))

  all_data_merged = load_custom_model_train(merged_data)
    # Store image number and merged data in lists
  merged_data_list.append(all_data_merged)

    # Print the shape of the merged data
  #print("Merged Data Shape:", merged_data_list)

  # Create a DataFrame with 'image number' and 'merged_data' columns
  df = pd.DataFrame({'all_sample_point_cloud_data': merged_data_list})
  preprocessed_data = df.to_numpy()

  return preprocessed_data


import os
import cv2
import numpy as np
from medpy.io import load as medpyload

""" def load_image(image_file_path):
    print("entered load_image")

    if os.path.exists(image_file_path):
        image_data, _ = medpyload(image_file_path)
    else:
        print(f"Image file '{image_file_path}' does not exist.")
        image_data = None
    
    print(image_data.shape)

    return image_data

 """


def predictions(image_file, model_weights_path, model):

    image_file_dict = create_image_data_dict(image_file)
    # Extract input features and point labels from the image data
    input_features = image_file_dict["input_features"]
    #point_labels = image_data["point_labels"] # image_data_dict["point_labels"]

    # Combine input_features and point_labels into a single array
    #combined_data = np.hstack((input_features))

    # Batch the combined_data
    batches = batch_data(input_features)

    # Load the pre-trained model with its weights
    model.load_weights(model_weights_path)

    # Initialize a list to store predictions and input points
    predictions = []
    input_points_list = []

    for idx, batch in enumerate(batches):
            X_test= batch[:, :3]
            X_data_test= X_test[np.newaxis, ...]
            # Perform prediction using the loaded model
            y_pred = model.predict(X_data_test)
            
            # Append the predictions and input points to their respective lists
            predictions.append(y_pred)
            input_points_list.append(X_data_test)
            
            label_predictions = [np.argmax(array, axis=-1) for array in predictions]
            pred_array = concatenate_xyz_and_labels(input_points_list,label_predictions)

    return pred_array


import os
import cv2
import numpy as np
from medpy.io import load as medpyload
import SimpleITK as sitk

def load_image(image_file_path):
    print("entered load_image")
    
    # Check if the file exists
    if not os.path.exists(image_file_path):
        print(f"Image file '{image_file_path}' does not exist.")
        return None

    # Check if it's a SimpleITK image (by checking its class)
    if isinstance(image_file_path, sitk.Image):
        print("Input is already a SimpleITK image.")
        return None

    # Check if it's an MHA image file
    if not image_file_path.endswith('.mha'):
        print(f"Input file '{image_file_path}' is not an MHA image.")
        return None

    # Load the image using MedPy
    try:
        image_data, _ = medpyload(image_file_path)
        if image_data is not None:
            print(f"Image data shape: {image_data.shape}")
            return image_data
    
    except Exception as e:
        print(f"Error loading the image: {str(e)}")
        return None
    


#image_file_path = "/home/golubeka/TopCoW Challenge/TopCoW-Challenge/TopCoW_Algo_Submission/test/input/images/head-ct-angio/uuid_of_ct_whole_066.mha"
#image_data = load_image(image_file_path)
