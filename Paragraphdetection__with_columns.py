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

def inpaint_paragraphs(img_path, pipeline, horizontal_threshold=100):
    img = preprocess_image(img_path)
    img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    prediction_groups = pipeline.recognize([img])

    for word, box in prediction_groups[0]:
        top_left = tuple(map(int, box[0]))
        bottom_right = tuple(map(int, box[2]))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Green boxes for words

    median_height = get_median_line_height([box for _, box in prediction_groups[0]])

    paragraphs = []
    current_paragraph = []
    threshold_y = 1.5 * median_height

    # Sort words by y-coordinates (top to bottom) for consistent paragraph detection
    words = sorted(prediction_groups[0], key=lambda x: x[1][0][1])

    for i, (word, box) in enumerate(words):
        if not current_paragraph:
            current_paragraph.append(box)
        else:
            last_word_box = current_paragraph[-1]
            vertical_distance = get_baseline_distance(last_word_box, box)
            horizontal_distance = get_horizontal_distance(last_word_box, box)

            # Check for column change by horizontal distance
            if horizontal_distance > horizontal_threshold:
                # New column detected, end the current paragraph
                paragraphs.append(current_paragraph)
                current_paragraph = [box]
            elif vertical_distance > threshold_y:
                # New paragraph within the same column
                paragraphs.append(current_paragraph)
                current_paragraph = [box]
            else:
                current_paragraph.append(box)

    if current_paragraph:
        paragraphs.append(current_paragraph)

    paragraph_coords = []
    for paragraph in paragraphs:
        min_x = min(coords[0][0] for coords in paragraph)
        max_x = max(coords[1][0] for coords in paragraph)
        min_y = min(coords[0][1] for coords in paragraph)
        max_y = max(coords[2][1] for coords in paragraph)
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
