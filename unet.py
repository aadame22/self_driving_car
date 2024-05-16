import matplotlib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

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

        # Map mask values to class indices
        mapped_mask = tf.map_fn(map_to_indices, mask, dtype=tf.int32)

        # Perform one-hot encoding
        num_classes = len(class_labels)
        mask_one_hot = tf.one_hot(mapped_mask, depth=num_classes)
        
        # Explicitly set the shape of the mask tensor
        mask_one_hot.set_shape([image_height, image_width, num_classes])  # Set the shape based on your expected shape
        
        return image, mask_one_hot
    else:
        return image


# Function to create dataset from image and mask paths
def create_dataset(image_paths, mask_paths=None, batch_size=32, shuffle=True):
    if mask_paths:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(load_and_preprocess_image)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Get list of image and mask paths
image_paths = [os.path.join(train_image_dir, filename) for filename in os.listdir(train_image_dir)]
mask_paths = [os.path.join(train_mask_dir, filename) for filename in os.listdir(train_mask_dir)]

val_paths = [os.path.join(val_image_dir, filename) for filename in os.listdir(val_image_dir)]
val_masks = [os.path.join(val_mask_dir, filename) for filename in os.listdir(val_mask_dir)]

test_paths = [os.path.join(test_image_dir, filename) for filename in os.listdir(test_image_dir)]
test_masks = [os.path.join(test_mask_dir, filename) for filename in os.listdir(test_mask_dir)]

## Split dataset into train, validation, and test sets
train_size = int(0.8 * len(image_paths))
val_size = int(0.15 * len(val_paths))
test_size = len(test_paths) - train_size - val_size

train_image_paths = image_paths[:train_size]
train_mask_paths = mask_paths[:train_size]

val_image_paths = val_paths[:val_size]
val_mask_paths = val_masks[:val_size]

test_image_paths = test_paths[:test_size]
test_mask_paths = test_paths[:test_size]

# Create datasets
train_dataset = create_dataset(train_image_paths, train_mask_paths)
val_dataset = create_dataset(val_image_paths, val_mask_paths)
test_dataset = create_dataset(test_image_paths, test_mask_paths)

# Define the UNet model architecture for binary segmentation
def unet_multi(input_shape, num_classes):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(256, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(128, 2, activation='relu', padding='same')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(merge7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(64, 2, activation='relu', padding='same')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(merge8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(32, 2, activation='relu', padding='same')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(merge9)
    
    # Output layer with sigmoid activation for binary classification
    outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create an instance of the UNet model for binary segmentation
model_multi= unet_multi(input_shape, num_classes)
# Compile the model for binary segmentation
model_multi.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[mean_iou, 'accuracy'])

# Train the model for binary segmentation
history_multi = model_multi.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset
)

# Evaluate the binary segmentation model
test_results = model_multi.evaluate(test_dataset)

# Load and preprocess an example input image
input_image_path = "temp/test/berlin_000000_000019_.png"
input_image = load_and_preprocess_image(input_image_path)  # Use your preprocessing function

# Reshape input image to represent a batch of size 1
input_image = tf.expand_dims(input_image, axis=0)

# Perform inference on the input image
predictions = model_multi.predict(input_image)

# Assuming `predictions` contains your prediction values
threshold = 0.5  # You can adjust this threshold as needed
segmentation_mask = np.argmax(predictions, axis=-1)  # Get the index of the class with the highest probability
segmentation_mask = np.where(predictions.max(axis=-1) > threshold, segmentation_mask, 0)  # Apply threshold

class_mapping = {
    0: "background",
    1: "car",
    2: "person",
    3: "road"
}
# Replace integer values with class labels
segmentation_mask_labels = np.vectorize(class_mapping.get)(segmentation_mask)
# Define a custom colormap for each class
colors = {
    'background': (0, 0, 0),   # Black
    'car': (0, 0, 142),         # Red
    'person': (220, 250, 60),      # Green
    'road': (128, 64, 128)         # Blue
}

# Convert color values to range [0, 1] and create a colormap
color_map = [(key, tuple(value / 255.0 for value in rgb)) for key, rgb in colors.items()]
cmap = matplotlib.colors.ListedColormap([color[1] for color in color_map])

# Visualize the input image and predicted mask
plt.figure(figsize=(10, 5))

# Input image
plt.subplot(1, 2, 1)
plt.imshow(input_image[0])
plt.title("Input Image")
plt.axis('off')

# Predicted mask
plt.subplot(1, 2, 2)
predicted_mask = np.squeeze(predictions, axis=0)  # Remove batch dimension
if len(segmentation_mask.shape) > 2:
    segmentation_mask = np.squeeze(segmentation_mask, axis=0)  # Remove batch dimension if it exists
plt.imshow(segmentation_mask, cmap=cmap, vmin=0, vmax=len(colors)-1)  # Use custom colormap
plt.title("Predicted Mask")
plt.axis('off')

plt.show()

# Plot training & validation accuracy values
plt.plot(history_multi.history['accuracy'])
plt.plot(history_multi.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation IoU values
plt.plot(history_multi.history['mean_iou'])
plt.plot(history_multi.history['val_mean_iou'])
plt.title('Mean IoU')
plt.ylabel('IoU')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_multi.history['loss'])
plt.plot(history_multi.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
