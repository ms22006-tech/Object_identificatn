import numpy as np
import cv2
import random
import os
import pandas as pd
import shutil

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def create_shape(image, shape_type, shape_size, x_center, y_center, shape_color):
    if shape_type == 'circle':
        cv2.circle(image, (x_center, y_center), shape_size, shape_color, -1)
    elif shape_type == 'square':
        cv2.rectangle(image, (x_center - shape_size // 2, y_center - shape_size // 2),
                      (x_center + shape_size // 2, y_center + shape_size // 2), shape_color, -1)
    elif shape_type == 'triangle':
        points = np.array([[x_center, y_center - shape_size],
                           [x_center - shape_size, y_center + shape_size],
                           [x_center + shape_size, y_center + shape_size]], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(image, [points], color=shape_color)
    elif shape_type == 'hexagon':
        points = np.array([[x_center + shape_size * np.cos(np.pi * 2 * i / 6), 
                            y_center + shape_size * np.sin(np.pi * 2 * i / 6)] for i in range(6)], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(image, [points], color=shape_color)
    elif shape_type == 'pentagon':
        points = np.array([[x_center + shape_size * np.cos(np.pi * 2 * i / 5), 
                            y_center + shape_size * np.sin(np.pi * 2 * i / 5)] for i in range(5)], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(image, [points], color=shape_color)

def random_dark_color():
    return tuple(random.randint(0, 100) for _ in range(3))

def is_non_overlapping(existing_shapes, x_center, y_center, shape_size):
    for (ex_x, ex_y, ex_size) in existing_shapes:
        if ((x_center - ex_x) ** 2 + (y_center - ex_y) ** 2) ** 0.5 < (shape_size + ex_size + 10):
            return False
    return True

def generate_image_with_shapes(image_size, shape_types, num_shapes=1):
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240
    shape_names = []
    existing_shapes = []
    
    for _ in range(num_shapes):
        shape_type = random.choice(shape_types)  
        placed = False
        attempts = 0
        while not placed and attempts < 100:  
            shape_size = random.randint(20, 30)  
            x_center = random.randint(shape_size + 5, image_size[0] - shape_size - 5)
            y_center = random.randint(shape_size + 5, image_size[1] - shape_size - 5)
            if is_non_overlapping(existing_shapes, x_center, y_center, shape_size):
                shape_color = random_dark_color()
                create_shape(image, shape_type, shape_size, x_center, y_center, shape_color)
                shape_names.append(shape_type)
                existing_shapes.append((x_center, y_center, shape_size))
                placed = True
            attempts += 1
    return image, shape_names

def generate_and_save_images(num_images, output_dir, labels_file):
    clear_directory(output_dir)
    labels_dir = os.path.dirname(labels_file)
    os.makedirs(labels_dir, exist_ok=True)

    all_labels = []
    shape_types = ['circle', 'square', 'triangle', 'hexagon', 'pentagon']
    
    num_multiple_shapes = int(num_images * 0.2)

    for i in range(num_images):
        if i < num_multiple_shapes:
            num_shapes = random.randint(2, 5)
        else:
            num_shapes = 1  
        image, shape_names = generate_image_with_shapes((128, 128), shape_types, num_shapes=num_shapes)
        
        img_name = f"{output_dir}/image_{i+1}.png"
        cv2.imwrite(img_name, image)
        
        all_labels.append({
            'filename': f"image_{i+1}.png",
            'shapes': ', '.join(shape_names)
        })
    
    df = pd.DataFrame(all_labels)
    df.to_csv(labels_file, index=False)

output_dir = 'colorful_images' 
labels_file = 'colorful_labels/labels.csv'

generate_and_save_images(2500, output_dir, labels_file)

print("Image generation complete and labels saved to CSV!")

