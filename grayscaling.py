import os
import cv2
import shutil
import numpy as np

def clear_directory(directory):

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def convert_to_black_and_white(input_dir, output_dir):

    clear_directory(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            image = cv2.imread(input_path)
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
            
            final_image = cv2.bitwise_not(binary_image)
            
            cv2.imwrite(output_path, final_image)

    print(f"Conversion complete. Black-and-white images saved to '{output_dir}'.")

# Specify directories
input_dir = 'colorful_images'  
output_dir = 'grayscaled_images'  

# Convert the images
convert_to_black_and_white(input_dir, output_dir)
