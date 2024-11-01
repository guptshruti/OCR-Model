import os
import cv2
import numpy as np
import keras_ocr

def preprocess_image(img_path):
    """Preprocess the image for better detection."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))  # Resize while maintaining aspect ratio
    return img

def get_median_distance(boxes, axis=0):
    """Calculate the median distance between boxes along a specified axis (0 for vertical, 1 for horizontal)."""
    distances = []
    for i in range(len(boxes) - 1):
        if axis == 0:  # Vertical distance
            distance = abs(boxes[i][0][1] - boxes[i + 1][0][1])  # y-coordinates
        else:  # Horizontal distance
            distance = abs(boxes[i][1][0] - boxes[i + 1][0][0])  # x-coordinates
        distances.append(distance)
    return np.median(distances) if distances else 0

def draw_bounding_boxes(image, paragraphs):
    """Draw bounding boxes around detected paragraphs on the image."""
    for paragraph in paragraphs:
        min_x = min(box[0][0] for box in paragraph)
        max_x = max(box[1][0] for box in paragraph)
        min_y = min(box[0][1] for box in paragraph)
        max_y = max(box[2][1] for box in paragraph)

        # Draw rectangle around the paragraph
        cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)  # Green box
    return image

def detect_paragraphs_columns(img_path, pipeline):
    """Detect paragraphs and columns based on distances between word boxes."""
    img = preprocess_image(img_path)
    prediction_groups = pipeline.recognize([img])

    # Extract word boxes
    boxes = [box for _, box in prediction_groups[0]]

    # Calculate median distances
    median_vertical_distance = get_median_distance(boxes, axis=0)
    median_horizontal_distance = get_median_distance(boxes, axis=1)

    threshold_y = median_vertical_distance * 1.5  # Vertical threshold for paragraphs
    threshold_x = median_horizontal_distance * 0.5  # Horizontal threshold for columns

    columns = []
    current_column = []
    paragraphs = []
    current_paragraph = []

    for i, (word, box) in enumerate(prediction_groups[0]):
        if not current_column:
            current_column.append(box)
            current_paragraph.append(box)
        else:
            last_box_column = current_column[-1]
            last_box_paragraph = current_paragraph[-1]

            vertical_distance = box[0][1] - last_box_paragraph[2][1]  # Start of current box - End of last box in paragraph
            horizontal_distance = box[0][0] - last_box_column[1][0]  # Start of current box - End of last box in column

            if horizontal_distance > threshold_x:
                # Start a new column if horizontal distance exceeds threshold
                if current_column:
                    columns.append(current_column)
                current_column = [box]
                current_paragraph = [box]
            elif vertical_distance > threshold_y:
                # Start a new paragraph within the same column
                paragraphs.append(current_paragraph)
                current_paragraph = [box]
                current_column.append(box)
            else:
                # Continue in the same column and paragraph
                current_paragraph.append(box)
                current_column.append(box)

    if current_paragraph:
        paragraphs.append(current_paragraph)  # Add last paragraph
    if current_column:
        columns.append(current_column)  # Add last column

    return img, paragraphs, columns

def save_paragraph_images(img, paragraphs, output_folder):
    """Save images of detected paragraphs."""
    for i, paragraph in enumerate(paragraphs):
        min_x = min(box[0][0] for box in paragraph)
        max_x = max(box[1][0] for box in paragraph)
        min_y = min(box[0][1] for box in paragraph)
        max_y = max(box[2][1] for box in paragraph)

        cropped_img = img[int(min_y):int(max_y), int(min_x):int(max_x)]
        cv2.imwrite(f"{output_folder}/paragraph_{i + 1}.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

def paragraph_detection(input_image, output_folder):
    """Main function to detect paragraphs and columns in the document."""
    # Initialize the keras-ocr pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # Process the image to get paragraph and column coordinates
    img_paragraphs, paragraphs, columns = detect_paragraphs_columns(input_image, pipeline)

    # Draw bounding boxes on the original image
    img_with_boxes = draw_bounding_boxes(img_paragraphs.copy(), paragraphs)

    # Save the output image with bounding boxes
    output_image_path = f"{output_folder}/bounding_boxes.png"
    cv2.imwrite(output_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

    # Save paragraph images
    save_paragraph_images(img_paragraphs, paragraphs, output_folder)

    print("Paragraph images and bounding boxes saved to:", output_folder)


if __name__ == "__main__":
    input_image_path = "/home/azureuser/lekhaanuvaad_processing/Test_images/test_english_final/2column.png"  # Update with your image path
    output_folder = "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/output_column"  # Update with your output folder path
    paragraph_detection(input_image_path, output_folder)
