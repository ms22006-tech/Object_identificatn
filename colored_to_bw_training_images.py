import os
import cv2
import numpy as np

input_dir = 'colorful_images'
output_dir = 'bw_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def detect_boundaries_with_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    bw_image = cv2.bitwise_not(dilated_edges)
    return bw_image

for filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path)
    bw_image = detect_boundaries_with_canny(image)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, bw_image)

print("Conversion to high-accuracy black-and-white images with visible boundaries is complete!")
