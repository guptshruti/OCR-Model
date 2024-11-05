import cv2
import pytesseract
from pytesseract import Output
import numpy as np
from sklearn.cluster import DBSCAN

def preprocess_image(img_path):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))  # Resize while maintaining aspect ratio
    return img

def detect_text(img):
    # Use Tesseract to detect text boxes
    custom_config = r'--oem 3 --psm 1'  # Page segmentation mode 1 (automatic layout analysis)
    boxes = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
    return boxes

def group_paragraphs(boxes, hor_threshold, ver_threshold):
    # Extract word coordinates and positions
    word_positions = []
    for i in range(len(boxes['text'])):
        if int(boxes['conf'][i]) > 60:  # Filtering low confidence detections
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
            word_positions.append((boxes['text'][i], (x, y, w, h)))

    # Calculate center points for clustering
    centers = np.array([[x + w / 2, y + h / 2] for _, (x, y, w, h) in word_positions])

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=min(hor_threshold, ver_threshold), min_samples=1).fit(centers)
    labels = clustering.labels_

    # Group words by paragraph clusters
    paragraphs = {}
    for idx, label in enumerate(labels):
        if label not in paragraphs:
            paragraphs[label] = []
        paragraphs[label].append(word_positions[idx])

    # Sort paragraphs by vertical position (for top-down reading order)
    sorted_paragraphs = sorted(paragraphs.values(), key=lambda para: min(word[1][1] for word in para))
    return sorted_paragraphs

def draw_paragraph_bounding_boxes(image, paragraphs):
    for paragraph in paragraphs:
        min_x = min(x for _, (x, y, w, h) in paragraph)
        max_x = max(x + w for _, (x, y, w, h) in paragraph)
        min_y = min(y for _, (x, y, w, h) in paragraph)
        max_y = max(y + h for _, (x, y, w, h) in paragraph)

        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    return image

def calculate_adaptive_thresholds(boxes):
    # Calculate average distances for thresholding based on layout
    horizontal_distances = []
    vertical_distances = []
    
    for i, (x, y, w, h) in enumerate(zip(boxes['left'], boxes['top'], boxes['width'], boxes['height'])):
        for j in range(i + 1, len(boxes['left'])):
            other_x, other_y, other_w, other_h = boxes['left'][j], boxes['top'][j], boxes['width'][j], boxes['height'][j]
            horizontal_distances.append(abs((x + w/2) - (other_x + other_w/2)))
            vertical_distances.append(abs((y + h/2) - (other_y + other_h/2)))

    hor_threshold = np.percentile(horizontal_distances, 25) * 0.75
    ver_threshold = np.percentile(vertical_distances, 25) * 1.2
    return hor_threshold, ver_threshold

def detect_paragraphs(input_image_path, output_folder):
    # Load image and detect text boxes
    img = preprocess_image(input_image_path)
    boxes = detect_text(img)

    # Calculate adaptive thresholds
    hor_threshold, ver_threshold = calculate_adaptive_thresholds(boxes)

    # Group words into paragraphs
    paragraphs = group_paragraphs(boxes, hor_threshold, ver_threshold)

    # Draw bounding boxes around detected paragraphs
    img_with_boxes = draw_paragraph_bounding_boxes(img.copy(), paragraphs)

    # Save the result
    output_path = os.path.join(output_folder, "tesseract_paragraph_bounding_boxes.png")
    cv2.imwrite(output_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"Paragraph bounding boxes saved to: {output_path}")

if __name__ == "__main__":
    input_image_path = "/path/to/your/input_image.png"
    output_folder = "/path/to/output_folder"
    detect_paragraphs(input_image_path, output_folder)
