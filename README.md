# Object_identificatn
-Manish Kaushik(MS22043)
-Apoorva Thapliyal (MS22006)
-Akshat Saini (Ms22085)
**Project Goal and Challenge**
The primary goal was to build a robust model capable of simultaneously classifying multiple shapes within a single input image.

**Problem:** Images are synthetically generated with random colors and sizes, featuring anywhere from 1 to 5 overlapping shapes.

Solution: A custom CNN architecture trained using the Sigmoid activation function on the output layer and Binary Cross-Entropy loss, which is the standard approach for multi-label tasks.
Model Architecture and Methodology
1. Dataset Generation and PreprocessingThe entire dataset was custom-generated to simulate a challenging, complex visual environment.
   **colorful_images.py** Generates 2,989 colorful images with 1 to 5 overlapping shapes.colorful_images/ & colorful_labels/labels.csvgrayscaling.pyConverts colorful images to $128 \times 128$ grayscale, which simplifies the recognition task for the CNN.grayscaled_images/
2.** CNN Architecture (training_model.py)**
   The model uses a standard sequential CNN for feature extraction:Input Shape: $128 \times 128 \times 1$ (Grayscale).Layers: 3 Convolutional blocks (Conv2D + MaxPooling2D) followed by   Flatten, a Dense hidden layer with Dropout (0.5) for regularization, and the output layer.Output Layer: 5 neurons (one for each shape class) with Sigmoid activation.Optimization: Adam optimizer with Binary Cross-Entropy loss.
   
   
