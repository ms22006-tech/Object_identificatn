import cv2
import numpy as np
import os
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(image_folder, label_csv):
    images = []
    labels = []
    
    
    with open(label_csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            image_filename = row[0]
            shape_labels = row[1].split(", ")
            image_path = os.path.join(image_folder, image_filename)
            
            
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(shape_labels)
    
    images = np.array(images)
    images = images.reshape(-1, 128, 128, 1)  
    images = images / 255.0  
    
    
    all_shapes = ['circle', 'square', 'triangle', 'pentagon', 'hexagon']
    label_encoded = []
    for label in labels:
    
        encoded = np.zeros(len(all_shapes))
        for shape in label:
            encoded[all_shapes.index(shape)] = 1
        label_encoded.append(encoded)
    
    label_encoded = np.array(label_encoded)
    
    return images, label_encoded

image_folder = 'grayscaled_images'  
label_csv = 'colorful_labels/labels.csv'  

images, labels = load_data(image_folder, label_csv)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,   
    width_shift_range=0.2,
    height_shift_range=0.2,  
    shear_range=0.2,   
    zoom_range=0.2,    
    horizontal_flip=True,  
    fill_mode='nearest'  
)

datagen.fit(X_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.BatchNormalization(),  
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(5, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.h5')

if os.path.exists(checkpoint_path):
    try:
        model.load_weights(checkpoint_path, by_name=True)  
        print(f"Resumed training from checkpoint: {checkpoint_path}")
    except ValueError as e:
        print(f"Error loading weights: {e}")
        print("Starting training from scratch.")
else:
    print("No checkpoint found, starting training from scratch.")

early_stopping = EarlyStopping(
    monitor='val_loss',   
    patience=5,           
    restore_best_weights=True,
    verbose=1
)

checkpoint_callback = ModelCheckpoint(
    checkpoint_path, 
    save_best_only=True,  
    save_weights_only=True,  
    monitor='val_loss',  
    verbose=1
)

model.fit(
    datagen.flow(X_train, y_train, batch_size=64), 
    epochs=50, 
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_callback, early_stopping]
)

model.save('shape_recognition_model_1.h5')

print(f"Model and weights saved to: {checkpoint_path}")
