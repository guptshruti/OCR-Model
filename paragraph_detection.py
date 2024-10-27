import os
import argparse
import keras_ocr
import cv2
import numpy as np
import shutil

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def get_median_line_height(boxes):
    heights = [abs(box[0][1] - box[2][1]) for box in boxes]
    return np.median(heights)

def get_baseline_distance(box1, box2):
    """Calculate the vertical distance between the baselines of two boxes."""
    return abs(box1[3][1] - box2[3][1])

def inpaint_paragraphs(img_path, pipeline):
    img = preprocess_image(img_path)
    # img_original = keras_ocr.tools.read(img_path)
    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    prediction_groups = pipeline.recognize([img])

    # Visualize word-level bounding boxes

    # for word, box in prediction_groups[0]:
    #     top_left = tuple(map(int, box[0]))
    #     bottom_right = tuple(map(int, box[2]))
    #     cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Green boxes for words

    # Get the average line height dynamically
    median_height = get_median_line_height([box for _, box in prediction_groups[0]])

    # Grouping words into paragraphs with baseline alignment and proximity checks
    paragraphs = []
    current_paragraph = []
    threshold_y = 1.5 * median_height  # Adjust threshold based on dynamic line height

    for word, box in prediction_groups[0]:
        if not current_paragraph:
            current_paragraph.append(box)
        else:
            # Check vertical distance and baseline consistency
            last_word_box = current_paragraph[-1]
            vertical_distance = get_baseline_distance(last_word_box, box)

            if vertical_distance > threshold_y:
                paragraphs.append(current_paragraph)
                current_paragraph = [box]
            else:
                current_paragraph.append(box)

    if current_paragraph:
        paragraphs.append(current_paragraph)

  
    paragraph_coords = []
    # Draw bounding boxes for paragraphs
    for paragraph in paragraphs:
        min_x = min(coords[0][0] for coords in paragraph)
        max_x = max(coords[1][0] for coords in paragraph)
        min_y = min(coords[0][1] for coords in paragraph)
        max_y = max(coords[2][1] for coords in paragraph)
        cv2.rectangle(img_original, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 1)  # Red boxes for paragraphs
        # Convert coordinates to integers and format as a simple list
        coords = ((int(min_x), int(min_y)), (int(max_x), int(max_y)))
        paragraph_coords.append(coords)
    

    return img_original, paragraph_coords

def save_paragraph_images(img, paragraphs, output_folder):
    # Delete the folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Remove the entire folder
    os.makedirs(output_folder)  # Create a new folder
    
    for i, (min_coords, max_coords) in enumerate(paragraphs):
        # Crop the image using the bounding box coordinates
        cropped_img = img[int(min_coords[1]):int(max_coords[1]), int(min_coords[0]):int(max_coords[0])]
        # Save the cropped image
        cv2.imwrite(os.path.join(output_folder, f'paragraph_{i + 1}.png'), cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

def paragraph_detection(input_image, output_folder):
    # Initialize the keras-ocr pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Process the image to get paragraph coordinates
    img_paragraphs, paragraph_coords = inpaint_paragraphs(input_image, pipeline)

    # Save the output image with bounding boxes (optional)
    output_image_path = os.path.join(output_folder, 'paragraph_bounding_boxes.png')
    cv2.imwrite(output_image_path, cv2.cvtColor(img_paragraphs, cv2.COLOR_BGR2RGB))

    # Save paragraph coordinates to a text file
    coordinates_file_path = os.path.join(output_folder, 'paragraph_coordinates.txt')
    with open(coordinates_file_path, 'w') as f:
        for coords in paragraph_coords:
            # Ensure that each element in paragraph_coords is a tuple of two tuples
            if len(coords) == 2:
                f.write(f"{coords[0]}, {coords[1]}\n")
            else:
                print("Error: Incorrect coordinate format")

    # Create a folder for cropped paragraph images
    paragraph_images_folder = os.path.join(output_folder, 'paragraph_images')
    save_paragraph_images(img_paragraphs, paragraph_coords, paragraph_images_folder)

    print("Paragraph bounding boxes coordinates saved to:", coordinates_file_path)
    print("Cropped paragraph images saved to:", paragraph_images_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image to extract paragraph-level coordinates and images.')
    parser.add_argument('input_image', type=str, help='Path to the input image file')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where results will be stored')
    
    args = parser.parse_args()
    paragraph_detection(args.input_image, args.output_folder)
