import cv2
import numpy as np
import os

# Set input and output folder paths
input_folder = 'Input Files\single_thermal'
output_folder = 'Output Files\single_thermal'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over all files in input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        # Read the image
        a = cv2.imread(os.path.join(input_folder, file_name))
        # Define the sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # Apply the sharpening kernel to the image
        img = cv2.filter2D(a, -1, kernel)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image to detect white color
        mask = cv2.inRange(img, np.array([10,10, 10]), np.array([255, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours and draw a square around white color regions
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:  # filter out small regions
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 10 and h >10:  # filter out small rectangles
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)

        # Save the result in output folder
        cv2.imwrite(os.path.join(output_folder, file_name), img)

        # Display the result
        # cv2.imshow('White Color Detection with Squares', img)
        # cv2.waitKey(0)

# cv2.destroyAllWindows()