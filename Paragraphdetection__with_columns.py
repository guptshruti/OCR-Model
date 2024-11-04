import os
import cv2
import numpy as np
import keras_ocr
from sklearn.cluster import DBSCAN

def preprocess_image(img_path):
    """Preprocess the image for better detection."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))  # Resize while maintaining aspect ratio
    return img

def get_column_boundaries(img, min_gap_width=50):
    """Detect column boundaries using vertical projection profiling."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Sum intensities vertically to detect gaps
    vertical_projection = np.sum(binary, axis=0)
    column_boundaries = []
    in_column = False
    start = 0

    for i, val in enumerate(vertical_projection):
        if val > 0 and not in_column:
            start = i
            in_column = True
        elif val == 0 and in_column:
            if i - start > min_gap_width:
                column_boundaries.append((start, i))
            in_column = False

    if not column_boundaries:
        print("Warning: No column boundaries detected.")
    return column_boundaries

def extract_words_by_column_and_line(prediction_groups, column_boundaries, line_eps=20):
    """Separate words into columns and lines based on detected column boundaries."""
    columns = [[] for _ in range(len(column_boundaries))]

    for word, box in prediction_groups[0]:
        x_center = (box[0][0] + box[1][0]) / 2
        y_center = (box[0][1] + box[2][1]) / 2

        # Place word in the corresponding column based on x_center
        placed = False
        for i, (start, end) in enumerate(column_boundaries):
            if start <= x_center < end:
                columns[i].append((word, box, y_center))
                placed = True
                break
        if not placed:
            print(f"Warning: Word '{word}' at ({x_center}, {y_center}) does not fit into any column boundary.")

    # Further divide each column into lines using y-coordinates
    column_lines = []
    for col in columns:
        if not col:
            print("Skipping empty column.")
            continue  # Skip empty columns

        y_coords = np.array([word[2] for word in col]).reshape(-1, 1)
        if y_coords.size == 0:
            print("Warning: No words found in this column after filtering.")
            continue

        clustering = DBSCAN(eps=line_eps, min_samples=1).fit(y_coords)

        lines_in_column = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in lines_in_column:
                lines_in_column[label] = []
            lines_in_column[label].append(col[idx])

        # Sort lines within column by y-coordinates
        sorted_lines = sorted(lines_in_column.values(), key=lambda line: min(word[2] for word in line))
        column_lines.append(sorted_lines)

    return column_lines

def draw_bounding_boxes(image, column_lines):
    """Draw bounding boxes around detected words in each line and column."""
    for column in column_lines:
        for line in column:
            for word, box, _ in line:
                min_x = min(box[i][0] for i in range(4))
                max_x = max(box[i][0] for i in range(4))
                min_y = min(box[i][1] for i in range(4))
                max_y = max(box[i][1] for i in range(4))
                cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)  # Green box
    return image

def save_word_images(img, column_lines, output_folder):
    """Save images of detected words."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 1
    for column in column_lines:
        for line in column:
            for word, box, _ in line:
                min_x = min(box[i][0] for i in range(4))
                max_x = max(box[i][0] for i in range(4))
                min_y = min(box[i][1] for i in range(4))
                max_y = max(box[i][1] for i in range(4))

                cropped_img = img[int(min_y):int(max_y), int(min_x):int(max_x)]
                cv2.imwrite(f"{output_folder}/word_{count}.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                count += 1

def inpaint_paragraphs_and_columns(img_path, pipeline):
    """Detect paragraphs and columns in the document."""
    img = preprocess_image(img_path)
    prediction_groups = pipeline.recognize([img])

    # Detect column boundaries
    column_boundaries = get_column_boundaries(img)

    # Separate words by columns and lines
    column_lines = extract_words_by_column_and_line(prediction_groups, column_boundaries)

    return img, column_lines

def paragraph_detection(input_image, output_folder):
    """Main function to detect paragraphs in the document."""
    pipeline = keras_ocr.pipeline.Pipeline()

    # Process the image to get paragraph coordinates
    img_paragraphs, column_lines = inpaint_paragraphs_and_columns(input_image, pipeline)

    # Draw bounding boxes on the original image
    img_with_boxes = draw_bounding_boxes(img_paragraphs.copy(), column_lines)

    # Save the output image with bounding boxes
    output_image_path = f"{output_folder}/bounding_boxes.png"
    cv2.imwrite(output_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

    # Save each word as an image
    save_word_images(img_paragraphs, column_lines, output_folder)

    print("Word images and bounding boxes saved to:", output_folder)

if __name__ == "__main__":
    input_image_path = "/home/azureuser/lekhaanuvaad_processing/Test_images/Gazette_Page_09.jpg"  # Update with your image path
    output_folder = "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/output_column"  # Update with your output folder path
    paragraph_detection(input_image_path, output_folder)
