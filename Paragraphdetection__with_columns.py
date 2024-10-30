import cv2
import numpy as np
import os
import shutil
import keras_ocr

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def get_median_line_height(boxes):
    heights = [abs(box[0][1] - box[2][1]) for box in boxes]
    return np.median(heights)

def get_average_word_width(boxes):
    widths = [abs(box[1][0] - box[0][0]) for box in boxes]
    return np.median(widths)

def get_baseline_distance(box1, box2):
    """Calculate the vertical distance between the baselines of two boxes."""
    return abs(box1[3][1] - box2[3][1])

def get_horizontal_distance(box1, box2):
    """Calculate the horizontal distance between the rightmost point of box1 and the leftmost point of box2."""
    return abs(box1[1][0] - box2[0][0])  # Rightmost of box1 to leftmost of box2

def inpaint_paragraphs(img_path, pipeline):
    img = preprocess_image(img_path)
    prediction_groups = pipeline.recognize([img])

    # Visualize word-level bounding boxes
    for word, box in prediction_groups[0]:
        top_left = tuple(map(int, box[0]))
        bottom_right = tuple(map(int, box[2]))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Green boxes for words

    # Get the average line height dynamically
    median_height = get_median_line_height([box for _, box in prediction_groups[0]])
    average_word_width = get_average_word_width([box for _, box in prediction_groups[0]])

    # Grouping words into lines first
    lines = []
    current_line = []
    threshold_y = 1.5 * median_height  # Vertical threshold based on line height

    for word, box in prediction_groups[0]:
        if not current_line:
            current_line.append(box)
        else:
            last_word_box = current_line[-1]
            vertical_distance = get_baseline_distance(last_word_box, box)

            if vertical_distance > threshold_y:
                lines.append(current_line)
                current_line = [box]
            else:
                current_line.append(box)

    if current_line:
        lines.append(current_line)

    # Now group lines into paragraphs based on horizontal distances
    paragraphs = []
    current_paragraph = []
    threshold_x = 1.5 * average_word_width  # Horizontal threshold based on average word width

    for line in lines:
        if not current_paragraph:
            current_paragraph.append(line)
        else:
            last_line = current_paragraph[-1]
            last_word_box = last_line[-1]  # Get the last word box of the last line
            first_word_box = line[0][0]  # Get the first word box of the current line

            # Debugging: Print the bounding boxes
            print("Last word box:", last_word_box)
            print("First word box:", first_word_box)

            # Ensure the boxes are in the expected format
            if isinstance(last_word_box, np.ndarray) and isinstance(first_word_box, np.ndarray):
                horizontal_distance = get_horizontal_distance(last_word_box, first_word_box)

                if horizontal_distance > threshold_x:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
            else:
                # If first_word_box is a single point, create a bounding box around it
                if isinstance(first_word_box, np.ndarray) and first_word_box.shape == (2,):
                    # Create a bounding box with a small width and height
                    first_word_box = np.array([[first_word_box[0], first_word_box[1]],
                                                [first_word_box[0] + 1, first_word_box[1]],
                                                [first_word_box[0] + 1, first_word_box[1] + 20],
                                                [first_word_box[0], first_word_box[1] + 20]])

                print("Warning: Unexpected box format. Using modified first word box.")

                horizontal_distance = get_horizontal_distance(last_word_box, first_word_box)

                if horizontal_distance > threshold_x:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)

    if current_paragraph:
        paragraphs.append(current_paragraph)

    paragraph_coords = []
    # Draw bounding boxes for paragraphs
    for paragraph in paragraphs:
        min_x = min(coords[0][0] for line in paragraph for coords in line)
        max_x = max(coords[1][0] for line in paragraph for coords in line)
        min_y = min(coords[0][1] for line in paragraph for coords in line)
        max_y = max(coords[2][1] for line in paragraph for coords in line)
        cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)  # Red boxes for paragraphs
        # Convert coordinates to integers and format as a simple list
        coords = ((int(min_x), int(min_y)), (int(max_x), int(max_y)))
        paragraph_coords.append(coords)

    return img, paragraph_coords

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

    # Create a folder for cropped paragraph images
    paragraph_images_folder = os.path.join(output_folder, 'paragraph_images')
    save_paragraph_images(img_paragraphs, paragraph_coords, paragraph_images_folder)

    print("Cropped paragraph images saved to:", paragraph_images_folder)


if __name__ == "__main__":
    input_image = '/home/azureuser/lekhaanuvaad_processing/Test_images/Gazette_Page_01.jpg'  # Update this path
    output_folder = '/home/azureuser/lekhaanuvaad_processing/paragraph_detection/output_paragraph'  # Update this path
    paragraph_detection(input_image, output_folder)
