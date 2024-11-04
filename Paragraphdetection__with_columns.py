import os
import keras_ocr
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Initialize the keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

def extract_words(image):
    """Use keras-ocr to detect words in the image and return their bounding boxes."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction_groups = pipeline.recognize([image_rgb])
    word_boxes = []
    for words in prediction_groups:
        for word in words:
            box, text = word[1], word[0]
            x_min = int(min(box[:, 0]))
            y_min = int(min(box[:, 1]))
            width = int(max(box[:, 0]) - x_min)
            height = int(max(box[:, 1]) - y_min)
            word_boxes.append((x_min, y_min, width, height, text))
    return word_boxes

def calculate_distances(word_boxes):
    """Calculate horizontal and vertical distances between word boxes."""
    distances = []
    for i in range(len(word_boxes)):
        for j in range(i + 1, len(word_boxes)):
            x1, y1, w1, h1, _ = word_boxes[i]
            x2, y2, w2, h2, _ = word_boxes[j]
            h_dist = abs(x1 - (x2 + w2))
            v_dist = abs(y1 - y2)
            distances.append((i, j, h_dist, v_dist))
    return distances

def calculate_dynamic_thresholds(word_boxes):
    """Calculate dynamic horizontal and vertical thresholds based on word distances."""
    h_dists = []
    v_dists = []

    for i in range(len(word_boxes)):
        for j in range(i + 1, len(word_boxes)):
            x1, y1, w1, h1, _ = word_boxes[i]
            x2, y2, w2, h2, _ = word_boxes[j]
            h_dist = abs(x1 - (x2 + w2))
            v_dist = abs(y1 - y2)

            h_dists.append(h_dist)
            v_dists.append(v_dist)

    # Calculate thresholds
    h_threshold = np.percentile(h_dists, 75) if h_dists else 50
    v_threshold = np.percentile(v_dists, 75) if v_dists else 20

    return h_threshold, v_threshold

def cluster_words(word_boxes, distances, h_threshold, v_threshold):
    """Cluster words using DBSCAN based on calculated distances."""
    distance_matrix = []
    for _, _, h_dist, v_dist in distances:
        distance_matrix.append([h_dist, v_dist])
    
    if distance_matrix:
        clustering = DBSCAN(eps=h_threshold, min_samples=1).fit(distance_matrix)
        clusters = [[] for _ in range(max(clustering.labels_) + 1)]
        
        for idx, label in enumerate(clustering.labels_):
            clusters[label].append(distances[idx][0])  # Append the word index

        return clusters
    return []

def draw_bounding_boxes(image, word_boxes, clusters):
    """Draw bounding boxes for each cluster."""
    for cluster in clusters:
        x_min = min(word_boxes[i][0] for i in cluster)
        y_min = min(word_boxes[i][1] for i in cluster)
        x_max = max(word_boxes[i][0] + word_boxes[i][2] for i in cluster)
        y_max = max(word_boxes[i][1] + word_boxes[i][3] for i in cluster)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

def process_document(image_path, output_folder):
    """Main processing function for the document."""
    # Load the image
    image = cv2.imread(image_path)

    # Step 1: Extract words using keras-ocr
    word_boxes = extract_words(image)

    # Step 2: Calculate distances
    distances = calculate_distances(word_boxes)

    # Step 3: Calculate dynamic thresholds
    horizontal_threshold, vertical_threshold = calculate_dynamic_thresholds(word_boxes)

    # Step 4: Cluster words based on thresholds
    clusters = cluster_words(word_boxes, distances, horizontal_threshold, vertical_threshold)

    # Step 5: Draw bounding boxes for clusters
    draw_bounding_boxes(image, word_boxes, clusters)

    # Save the result
    output_path = os.path.join(output_folder, 'processed_document.png')
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = 'path_to_your_document_image.jpg'  # Replace with your input image path
    output_folder = 'output'  # Specify your output folder here

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_document(image_path, output_folder)
