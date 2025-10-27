import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# --- Configuration (Must match training_model.py) ---
IMG_HEIGHT, IMG_WIDTH = 128, 128
SHAPE_CLASSES = ['circle', 'square', 'triangle', 'hexagon', 'pentagon']
images_dir = 'grayscaled_images'
labels_file = 'colorful_labels/labels.csv'
model_path = 'best_model.h5'
SEED = 42 # CRUCIAL: Must match training script

# --- Data Loading Function ---
def load_data(images_dir, labels_file):
    labels_df = pd.read_csv(labels_file)
    images = []
    labels = []

    for _, row in labels_df.iterrows():
        image_path = os.path.join(images_dir, row['filename'])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            continue
            
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0
        images.append(image.reshape(IMG_HEIGHT, IMG_WIDTH, 1))

        shapes = row['shapes'].split(', ')
        labels.append(shapes)

    # Return labels_df for filename retrieval (Fixes NameError)
    return np.array(images), labels, labels_df

# --- Main Execution ---
try:
    # 1. Load Data
    X, labels_list, labels_df = load_data(images_dir, labels_file)
    
    # 2. Binarize labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels_list)
    
    # 3. Split data using the SAME random_state and capture indices
    # We split X, y, and the list of original indices (range(len(X)))
    # to maintain alignment after shuffling.
    _, X_test, _, y_test, _, X_test_indices = train_test_split(
        X, y, range(len(X)), test_size=0.2, random_state=SEED
    )

    # 4. Load model
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    
    # 5. Make predictions (probabilities)
    y_pred = model.predict(X_test)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the data generation, grayscaling, and training steps were completed successfully.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()


# -----------------------------------------------------
# --- CORE EVALUATION ---
# -----------------------------------------------------

# 1. Convert prediction probabilities to binary predictions (threshold > 0.5)
y_pred_binary = (y_pred > 0.5).astype(int)

# Exact Match Ratio (Strict accuracy: every single label must be correct)
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy*100:.2f}% (Exact Match Ratio)")


# 2. ROC-AUC Score (Uses probabilities y_pred)
try:
    roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
    print(f"ROC-AUC Score (weighted): {roc_auc:.4f}")
except ValueError:
    print("Could not compute ROC-AUC.")


# 3. Classification Report (Uses binary predictions y_pred_binary)
class_report = classification_report(y_test, y_pred_binary, target_names=SHAPE_CLASSES)
print("\nClassification Report (Micro-Averages are best for multi-label):")
print(class_report)


# 4. Confusion Matrix (Single-Label View for visualization)
y_test_single = y_test.argmax(axis=1)
y_pred_single = y_pred.argmax(axis=1)

conf_matrix = confusion_matrix(y_test_single, y_pred_single)
print("\nConfusion Matrix (Based on Single Most Confident Label):")
print(conf_matrix)


# --- Plotting and Saving Results ---

# Generate the list of filenames and corresponding predicted shapes
predicted_labels_list = []
# Use the X_test_indices to correctly fetch the filenames from the original DataFrame
for i, row in enumerate(y_pred_binary):
    original_index = X_test_indices[i]
    filename = labels_df.iloc[original_index]['filename'] 
    predicted_shapes = mlb.inverse_transform(np.array([row]))[0]
    predicted_labels_list.append([filename, ', '.join(predicted_shapes)])

# Save the predicted labels with filenames
predicted_labels_df = pd.DataFrame(predicted_labels_list, columns=['filename', 'predicted_shapes'])
predicted_labels_file = 'predicted_labels_test_set.csv'
predicted_labels_df.to_csv(predicted_labels_file, index=False)
print(f"\nPredicted labels saved to {predicted_labels_file}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=SHAPE_CLASSES, yticklabels=SHAPE_CLASSES)
plt.title("Confusion Matrix (Single-Label View)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix_plot.png")

# Calculate and plot class-wise accuracy
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
plt.figure(figsize=(10, 6))
plt.bar(SHAPE_CLASSES, class_accuracies, color='skyblue')
plt.title("Class-wise Accuracy (Single-Label View)")
plt.xlabel("Shape Class")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("class_accuracy_plot.png")


print("\nSaved confusion_matrix_plot.png and class_accuracy_plot.png.")
