import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from keras.metrics import MeanIoU

# Define input shape and number of classes
input_shape = (240, 320, 3)  # Adjust based on your image dimensions
num_classes = 4  # Number of classes in the Cityscapes dataset

# Define dataset paths
dataset_dir = "temp"
train_image_dir = os.path.join(dataset_dir, "train")
train_mask_dir = os.path.join(dataset_dir, "train_labels")

val_image_dir = os.path.join(dataset_dir, "val")
val_mask_dir = os.path.join(dataset_dir, "val_labels")

test_image_dir = os.path.join(dataset_dir, "test")
test_mask_dir = os.path.join(dataset_dir, "test_labels")

# Define image dimensions
image_height = 240
image_width = 320

class_labels = {
    'background': 0,
    'car': 1,
     'person': 2,
    'road': 3
}

def mean_iou(y_true, y_pred):
    miou = MeanIoU(num_classes=num_classes)
    miou.update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    return miou.result()

def load_and_preprocess_image(image_path, mask_path=None):
    # Enable eager execution
    tf.config.run_functions_eagerly(True)
    
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    
    
    if mask_path is not None:
        # Define class labels and pixel values
        class_labels = {
            'background': 0,
            'car': 1,
            'person': 2,
            'road': 3
        }
        pixel_values = [0, 16, 84, 90]  # Example pixel values for each class
        class_indices = [class_labels['background'], class_labels['car'], class_labels['person'], class_labels['road']]

        # Create a lookup table using tf.lookup.StaticHashTable
        keys_tensor = tf.constant(pixel_values)
        values_tensor = tf.constant(class_indices)
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
            default_value=len(class_labels)  # Use len(class_labels) as the default value for unknown keys
        )


        # Define a function to map mask values to class indices
        def map_to_indices(pixel_value):
            return table.lookup(pixel_value)

        # Load and preprocess mask image
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)  # Assuming single-channel mask
        mask = tf.squeeze(mask)  # Remove singleton dimension if present
        mask = tf.cast(mask, tf.int32)  # Convert mask to integer type
        print(mask)
        # Map mask values to class indices
        mapped_mask = tf.map_fn(map_to_indices, mask, dtype=tf.int32)

        # Perform one-hot encoding
        num_classes = len(class_labels)
        mask_one_hot = tf.one_hot(mapped_mask, depth=num_classes)
        
        # Explicitly set the shape of the mask tensor
        mask_one_hot.set_shape([image_height, image_width, num_classes])  # Set the shape based on your expected shape

        print(mask_one_hot)
        
        return image, mask_one_hot
    else:
        return image
    
test_sample_image = "temp/train/aachen_000000_000019_.png"
test_sample_mask = "temp/train_labels/aachen_000000_000019_.png" 
load_and_preprocess_image(test_sample_image, test_sample_mask)

