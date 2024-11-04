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

    return column_boundaries

def extract_words_by_column(prediction_groups, column_boundaries):
    """Separate words into columns based on detected column boundaries."""
    columns = [[] for _ in range(len(column_boundaries))]

    for word, box in prediction_groups[0]:
        x_center = (box[0][0] + box[1][0]) / 2
        for i, (start, end) in enumerate(column_boundaries):
            if start <= x_center < end:
                columns[i].append(box)
                break

    return columns

def cluster_paragraphs(column_words, eps=20, min_samples=2):
    """Cluster words into paragraphs within each column using DBSCAN."""
    paragraphs = []
    for boxes in column_words:
        if not boxes:
            continue
        y_coords = np.array([box[0][1] for box in boxes]).reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)

        # Group words by cluster label
        paragraph_dict = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue  # Ignore noise
            if label not in paragraph_dict:
                paragraph_dict[label] = []
            paragraph_dict[label].append(boxes[idx])
        
        # Append paragraphs in current column
        paragraphs.extend(paragraph_dict.values())
    
    return paragraphs

def draw_bounding_boxes(image, paragraphs):
    """Draw bounding boxes around detected paragraphs on the image."""
    for paragraph in paragraphs:
        min_x = min(box[0][0] for box in paragraph)
        max_x = max(box[1][0] for box in paragraph)
        min_y = min(box[0][1] for box in paragraph)
        max_y = max(box[2][1] for box in paragraph)
        cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)  # Green box
    return image

def save_paragraph_images(img, paragraphs, output_folder):
    """Save images of detected paragraphs."""
    for i, paragraph in enumerate(paragraphs):
        min_x = min(box[0][0] for box in paragraph)
        max_x = max(box[1][0] for box in paragraph)
        min_y = min(box[0][1] for box in paragraph)
        max_y = max(box[2][1] for box in paragraph)

        cropped_img = img[int(min_y):int(max_y), int(min_x):int(max_x)]
        #cv2.imwrite(f"{output_folder}/paragraph_{i + 1}.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

def inpaint_paragraphs_and_columns(img_path, pipeline):
    """Detect paragraphs and columns in the document."""
    img = preprocess_image(img_path)
    prediction_groups = pipeline.recognize([img])

    # Detect column boundaries
    column_boundaries = get_column_boundaries(img)

    # Separate words by columns
    column_words = extract_words_by_column(prediction_groups, column_boundaries)

    # Detect paragraphs within each column
    paragraphs = cluster_paragraphs(column_words)

    return img, paragraphs

def paragraph_detection(input_image, output_folder):
    """Main function to detect paragraphs in the document."""
    pipeline = keras_ocr.pipeline.Pipeline()

    # Process the image to get paragraph coordinates
    img_paragraphs, paragraphs = inpaint_paragraphs_and_columns(input_image, pipeline)

    # Draw bounding boxes on the original image
    img_with_boxes = draw_bounding_boxes(img_paragraphs.copy(), paragraphs)

    # Save the output image with bounding boxes
    output_image_path = f"{output_folder}/bounding_boxes.png"
    cv2.imwrite(output_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

    # Save paragraph images
    save_paragraph_images(img_paragraphs, paragraphs, output_folder)

    print("Paragraph images and bounding boxes saved to:", output_folder)

if __name__ == "__main__":
    input_image_path = "/home/azureuser/lekhaanuvaad_processing/Test_images/Gazette_Page_09.jpg"  # Update with your image path
    output_folder = "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/output_column"  # Update with your output folder path
    paragraph_detection(input_image_path, output_folder)
