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
    return abs(box1[3][1] - box2[3][1])

def get_horizontal_distance(box1, box2):
    return abs(box1[1][0] - box2[0][0])

def inpaint_paragraphs(img_path, pipeline, horizontal_threshold=100, vertical_threshold_factor=1.5):
    img = preprocess_image(img_path)
    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    prediction_groups = pipeline.recognize([img])

    # Sort all detected words by their top-left y-coordinate for top-to-bottom ordering
    words = sorted(prediction_groups[0], key=lambda x: x[1][0][1])

    # Detect median height of a line for determining line and paragraph spacing
    median_height = get_median_line_height([box for _, box in prediction_groups[0]])
    vertical_threshold = vertical_threshold_factor * median_height

    paragraphs = []
    current_paragraph = []
    current_line = []
    current_column = []

    # Group words into lines
    for i, (word, box) in enumerate(words):
        if not current_line:
            current_line.append((word, box))
            continue
        
        last_word = current_line[-1][1]
        vertical_distance = get_baseline_distance(last_word, box)
        horizontal_distance = get_horizontal_distance(last_word, box)

        if horizontal_distance > horizontal_threshold:
            # New column starts
            paragraphs.append(current_column)
            current_column = []
            current_line = [(word, box)]
        elif vertical_distance > vertical_threshold:
            # New paragraph starts within the same column
            current_column.append(current_line)
            current_line = [(word, box)]
        else:
            # Continue current line
            current_line.append((word, box))
    
    # Add the last line and column to paragraphs
    if current_line:
        current_column.append(current_line)
    if current_column:
        paragraphs.append(current_column)

    # Draw bounding boxes for paragraphs
    paragraph_coords = []
    for column in paragraphs:
        for lines in column:
            min_x = min(word_box[1][0][0] for word_box in lines)
            max_x = max(word_box[1][1][0] for word_box in lines)
            min_y = min(word_box[1][0][1] for word_box in lines)
            max_y = max(word_box[1][2][1] for word_box in lines)
            cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)  # Red boxes for paragraphs
            coords = ((int(min_x), int(min_y)), (int(max_x), int(max_y)))
            paragraph_coords.append(coords)

    return img, paragraph_coords

def save_paragraph_images(img, paragraphs, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, (min_coords, max_coords) in enumerate(paragraphs):
        cropped_img = img[int(min_coords[1]):int(max_coords[1]), int(min_coords[0]):int(max_coords[0])]
        cv2.imwrite(os.path.join(output_folder, f'paragraph_{i + 1}.png'), cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

def paragraph_detection(input_image, output_folder):
    pipeline = keras_ocr.pipeline.Pipeline()

    img_paragraphs, paragraph_coords = inpaint_paragraphs(input_image, pipeline)

    output_image_path = os.path.join(output_folder, 'paragraph_bounding_boxes.png')
    cv2.imwrite(output_image_path, cv2.cvtColor(img_paragraphs, cv2.COLOR_BGR2RGB))

    coordinates_file_path = os.path.join(output_folder, 'paragraph_coordinates.txt')
    with open(coordinates_file_path, 'w') as f:
        for coords in paragraph_coords:
            if len(coords) == 2:
                f.write(f"{coords[0]}, {coords[1]}\n")
            else:
                print("Error: Incorrect coordinate format")

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
