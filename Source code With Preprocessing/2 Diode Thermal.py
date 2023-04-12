import cv2
import numpy as np
import glob
import os

input_folder = 'Input Files\diode_thermal'
output_folder = 'diode_thermal'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_path in glob.glob(os.path.join(input_folder, '*.jpg')):

    img = cv2.imread(file_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower = np.array([200, 200, 200], dtype=np.uint8)
    upper = np.array([255, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img, lower, upper)

    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    # Exclude upper-left quarter region
    height, width, _ = img.shape
    quarter_height = height // 2
    quarter_width = width // 2
    dilation[:quarter_height, :quarter_width] = 0

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = img.copy()
    # Exclude upper-left quarter region from contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < quarter_width and y < quarter_height:
            continue
        cv2.drawContours(img_contour, [cnt], -1, (0, 0, 0), thickness=2)

    # Add 3% border
    border_size = int(min(height, width) * 0.03)
    img_contour = cv2.copyMakeBorder(img_contour, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    cv2.imwrite(output_file_path, img_contour)