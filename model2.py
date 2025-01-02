# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Fix TensorFlow messages (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define the WDU-Net model
def deformable_conv_block(inputs, filters, kernel_size=3):
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
    return x

def weight_generation_block(encoder_features, decoder_features):
    # Align the channel dimensions of encoder and decoder features
    decoder_features = Conv2D(encoder_features.shape[-1], (1, 1), padding='same')(decoder_features)
    
    # Add the aligned tensors
    combined = tf.keras.layers.Add()([encoder_features, decoder_features])
    
    # Apply weights and return
    weights = tf.keras.layers.Activation('sigmoid')(combined)
    weighted_features = tf.keras.layers.Multiply()([encoder_features, weights])
    return weighted_features


def WDU_Net(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)
    
    # Encoder
    e1 = deformable_conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(e1)
    
    e2 = deformable_conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(e2)
    
    e3 = deformable_conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(e3)
    
    # Bottleneck
    b = deformable_conv_block(p3, 512)
    
    # Decoder
    d3 = UpSampling2D((2, 2))(b)
    d3 = weight_generation_block(e3, d3)
    d3 = deformable_conv_block(d3, 256)
    
    d2 = UpSampling2D((2, 2))(d3)
    d2 = weight_generation_block(e2, d2)
    d2 = deformable_conv_block(d2, 128)
    
    d1 = UpSampling2D((2, 2))(d2)
    d1 = weight_generation_block(e1, d1)
    d1 = deformable_conv_block(d1, 64)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d1)
    
    model = Model(inputs, outputs, name="WDU-Net")
    return model

# Define the focal asymmetric similarity loss function
def focal_asymmetric_similarity_loss(y_true, y_pred, alpha=0.7, gamma=2, beta=1.5, lambda_=0.65):
    focal_loss = -alpha * ((1 - y_pred) ** gamma) * tf.math.log(y_pred + 1e-7)
    asymmetric_similarity = (1 + beta**2) * y_true * y_pred / (
        (1 + beta**2) * y_true * y_pred + beta**2 * (1 - y_pred) * y_true + y_pred * (1 - y_true) + 1e-7
    )
    return lambda_ * asymmetric_similarity + (1 - lambda_) * focal_loss

# Load and preprocess dataset
def load_custom_dataset(base_dir, target_size=(128, 128)):
    images = []
    masks = []
    
    for patient_dir in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        for nodule_dir in os.listdir(patient_path):
            nodule_path = os.path.join(patient_path, nodule_dir)
            images_dir = os.path.join(nodule_path, "images")
            masks_dir = os.path.join(nodule_path, "mask-0")

            for image_file in sorted(os.listdir(images_dir)):
                img_path = os.path.join(images_dir, image_file)
                mask_path = os.path.join(masks_dir, image_file)  # Assuming mask filenames match image filenames

                # Load and preprocess images
                img = load_img(img_path, target_size=target_size, color_mode="grayscale")
                img = img_to_array(img) / 255.0  # Normalize to [0, 1]

                # Load and preprocess masks
                mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
                mask = img_to_array(mask) / 255.0  # Normalize to [0, 1]

                images.append(img)
                masks.append(mask)
    
    return np.array(images), np.array(masks)

# Main script
if __name__ == "__main__":
    # Specify dataset path
    dataset_path = "/home/guest_miu/Lungtumor_ct/extracted_dataset/LIDC-IDRI-slices"

    # Load dataset
    images, masks = load_custom_dataset(dataset_path)
    print(f"Loaded {len(images)} images and {len(masks)} masks successfully!")

    # Split dataset into train and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Define and compile the model
    model = WDU_Net(input_shape=(128, 128, 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=focal_asymmetric_similarity_loss,
                  metrics=['accuracy'])
    print("Model compiled successfully!")

    # Train the model
    batch_size = 4
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(batch_size).shuffle(100).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model.fit(train_dataset, validation_data=val_dataset, epochs=2)

    # Save the trained model
    model.save("wdu_net_model.h5")
    print("Model saved successfully!")
