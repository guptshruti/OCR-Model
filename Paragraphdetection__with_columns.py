import os
import argparse
import cv2
import numpy as np
import shutil

def preprocess_image(img_path):
    # Load the image and convert to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Apply Gaussian blur to reduce noise and detail
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply adaptive threshold to separate text from background
    img_threshold = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
    return img_threshold

def find_paragraph_contours(img):
    # Create a structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Dilate the image to connect text blocks and form paragraph-like shapes
    dilated = cv2.dilate(img, kernel, iterations=2)
    # Find contours on the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size to exclude small bounding boxes
    paragraph_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    return paragraph_contours

def inpaint_paragraphs(img_path):
    # Preprocess the image
    img = preprocess_image(img_path)
    # Load the original image for drawing
    img_original = cv2.imread(img_path)
    
    # Detect paragraph contours
    paragraph_contours = find_paragraph_contours(img)
    
    paragraph_coords = []
    for cnt in paragraph_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        paragraph_coords.append(((x, y), (x + w, y + h)))
        # Draw bounding box around each detected paragraph
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red boxes for paragraphs

    return img_original, paragraph_coords

def save_paragraph_images(img, paragraphs, output_folder):
    # Delete the folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, (min_coords, max_coords) in enumerate(paragraphs):
        cropped_img = img[min_coords[1]:max_coords[1], min_coords[0]:max_coords[0]]
        cv2.imwrite(os.path.join(output_folder, f'paragraph_{i + 1}.png'), cropped_img)

def paragraph_detection(input_image, output_folder):
    # Process the image to get paragraph coordinates
    img_paragraphs, paragraph_coords = inpaint_paragraphs(input_image)

    # Save the output image with bounding boxes
    output_image_path = os.path.join(output_folder, 'paragraph_bounding_boxes.png')
    cv2.imwrite(output_image_path, img_paragraphs)

    # Save paragraph coordinates to a text file
    coordinates_file_path = os.path.join(output_folder, 'paragraph_coordinates.txt')
    with open(coordinates_file_path, 'w') as f:
        for coords in paragraph_coords:
            f.write(f"{coords[0]}, {coords[1]}\n")

    # Save cropped paragraph images
    paragraph_images_folder = os.path.join(output_folder, 'paragraph_images')
    save_paragraph_images(cv2.imread(input_image), paragraph_coords, paragraph_images_folder)

    print("Paragraph bounding boxes coordinates saved to:", coordinates_file_path)
    print("Cropped paragraph images saved to:", paragraph_images_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image to extract paragraph-level coordinates and images.')
    parser.add_argument('input_image', type=str, help='Path to the input image file')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where results will be stored')

    args = parser.parse_args()
    paragraph_detection(args.input_image, args.output_folder)
