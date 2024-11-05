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

def get_adaptive_thresholds(words_boxes):
    """Calculate adaptive thresholds by analyzing word distances."""
    horizontal_distances = []
    vertical_distances = []

    # Calculate pairwise distances between all words to adapt thresholds dynamically
    for i, (word, box) in enumerate(words_boxes):
        x_center, y_center = np.mean(box, axis=0)
        for j, (_, other_box) in enumerate(words_boxes):
            if i == j:
                continue
            other_x_center, other_y_center = np.mean(other_box, axis=0)
            horizontal_distances.append(abs(x_center - other_x_center))
            vertical_distances.append(abs(y_center - other_y_center))

    # Determine dynamic thresholds based on document spacing patterns
    hor_threshold = np.percentile(horizontal_distances, 20)
    ver_threshold = np.percentile(vertical_distances, 20)
    return hor_threshold, ver_threshold

def group_words_by_adaptive_clustering(prediction_groups, hor_threshold, ver_threshold):
    """Group words into clusters using adaptive DBSCAN thresholds."""
    words_data = [(word, box) for word, box in prediction_groups[0]]
    centers = np.array([np.mean(box, axis=0) for _, box in words_data])

    # DBSCAN with adaptive horizontal and vertical thresholds
    clustering = DBSCAN(eps=min(hor_threshold, ver_threshold), min_samples=1, metric='euclidean').fit(centers)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(words_data[idx])

    # Sort clusters by vertical position to handle multi-column layouts
    sorted_clusters = [sorted(cluster, key=lambda word: np.mean(word[1], axis=0)[1]) for cluster in clusters.values()]
    sorted_clusters.sort(key=lambda cluster: np.mean([np.mean(word[1], axis=0)[1] for word in cluster]))

    return sorted_clusters

def draw_paragraph_bounding_boxes(image, paragraphs):
    """Draw bounding boxes around each paragraph."""
    for paragraph in paragraphs:
        min_x = min(box[0][0] for _, box in paragraph)
        max_x = max(box[1][0] for _, box in paragraph)
        min_y = min(box[0][1] for _, box in paragraph)
        max_y = max(box[2][1] for _, box in paragraph)

        # Draw bounding box
        cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

    return image

def detect_paragraphs(input_image_path, output_folder):
    """Main function to detect paragraphs and draw bounding boxes."""
    pipeline = keras_ocr.pipeline.Pipeline()

    # Preprocess the image
    img = preprocess_image(input_image_path)
    predictions = pipeline.recognize([img])

    # Determine adaptive thresholds
    hor_threshold, ver_threshold = get_adaptive_thresholds(predictions[0])

    # Group words into clusters and paragraphs using adaptive clustering
    paragraphs = group_words_by_adaptive_clustering(predictions, hor_threshold, ver_threshold)

    # Draw bounding boxes around paragraphs
    img_with_boxes = draw_paragraph_bounding_boxes(img.copy(), paragraphs)

    # Save the result
    output_path = os.path.join(output_folder, "adaptive_paragraph_bounding_boxes.png")
    cv2.imwrite(output_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"Paragraph bounding boxes saved to: {output_path}")

if __name__ == "__main__":
    input_image_path = "/path/to/your/input_image.png"
    output_folder = "/path/to/output_folder"
    detect_paragraphs(input_image_path, output_folder)
