import os
import cv2
import numpy as np
import keras_ocr
from sklearn.cluster import DBSCAN

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))  # Resize while maintaining aspect ratio
    return img

def calculate_adaptive_thresholds(words_boxes):
    horizontal_distances = []
    vertical_distances = []

    for i, (word, box) in enumerate(words_boxes):
        x_center, y_center = np.mean(box, axis=0)
        for j, (_, other_box) in enumerate(words_boxes):
            if i == j:
                continue
            other_x_center, other_y_center = np.mean(other_box, axis=0)
            horizontal_distances.append(abs(x_center - other_x_center))
            vertical_distances.append(abs(y_center - other_y_center))

    hor_threshold = np.percentile(horizontal_distances, 25) * 0.8  # Dynamic scaling
    ver_threshold = np.percentile(vertical_distances, 25) * 0.8
    return hor_threshold, ver_threshold

def group_words(prediction_groups, hor_threshold, ver_threshold):
    words_data = [(word, box) for word, box in prediction_groups[0]]
    centers = np.array([np.mean(box, axis=0) for _, box in words_data])

    clustering = DBSCAN(eps=min(hor_threshold, ver_threshold), min_samples=1).fit(centers)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(words_data[idx])

    # Sort clusters and refine by column alignment, then split based on paragraph gap detection
    sorted_clusters = [sorted(cluster, key=lambda word: np.mean(word[1], axis=0)[1]) for cluster in clusters.values()]
    sorted_clusters.sort(key=lambda cluster: np.mean([np.mean(word[1], axis=0)[1] for word in cluster]))

    refined_clusters = []
    for cluster in sorted_clusters:
        if len(cluster) > 1:
            # Detect large gaps indicating paragraph breaks
            paragraph_groups = []
            current_paragraph = [cluster[0]]
            vertical_gaps = []
            for i in range(1, len(cluster)):
                prev_y = np.mean(cluster[i - 1][1], axis=0)[1]
                curr_y = np.mean(cluster[i][1], axis=0)[1]
                vertical_gap = abs(curr_y - prev_y)
                vertical_gaps.append(vertical_gap)

            # Compute dynamic paragraph threshold using mean + std dev
            mean_gap = np.mean(vertical_gaps)
            std_dev_gap = np.std(vertical_gaps)
            dynamic_ver_threshold = mean_gap + std_dev_gap  # Adjust with std deviation to adapt to spacing

            for i in range(1, len(cluster)):
                prev_y = np.mean(cluster[i - 1][1], axis=0)[1]
                curr_y = np.mean(cluster[i][1], axis=0)[1]
                vertical_gap = abs(curr_y - prev_y)

                # If vertical gap exceeds dynamic threshold, create new paragraph
                if vertical_gap > dynamic_ver_threshold:
                    paragraph_groups.append(current_paragraph)
                    current_paragraph = [cluster[i]]
                else:
                    current_paragraph.append(cluster[i])

            paragraph_groups.append(current_paragraph)
            refined_clusters.extend(paragraph_groups)
        else:
            refined_clusters.append(cluster)

    return refine_column_clusters(refined_clusters, hor_threshold)


def refine_column_clusters(clusters, hor_threshold):
    refined_clusters = []
    for cluster in clusters:
        if len(cluster) > 1:
            x_positions = [np.mean(word[1], axis=0)[0] for word in cluster]
            avg_x = np.mean(x_positions)
            left, right = [], []
            for word, box in cluster:
                if np.mean(box, axis=0)[0] < avg_x:
                    left.append((word, box))
                else:
                    right.append((word, box))
            if left and right:
                refined_clusters.append(left)
                refined_clusters.append(right)
            else:
                refined_clusters.append(cluster)
        else:
            refined_clusters.append(cluster)
    return refined_clusters

def draw_paragraph_bounding_boxes(image, paragraphs):
    for paragraph in paragraphs:
        min_x = min(box[0][0] for _, box in paragraph)
        max_x = max(box[1][0] for _, box in paragraph)
        min_y = min(box[0][1] for _, box in paragraph)
        max_y = max(box[2][1] for _, box in paragraph)

        cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

    return image

def detect_paragraphs(input_image_path, output_folder):
    pipeline = keras_ocr.pipeline.Pipeline()

    img = preprocess_image(input_image_path)
    predictions = pipeline.recognize([img])

    hor_threshold, ver_threshold = calculate_adaptive_thresholds(predictions[0])

    paragraphs = group_words(predictions, hor_threshold, ver_threshold)

    img_with_boxes = draw_paragraph_bounding_boxes(img.copy(), paragraphs)

    output_path = os.path.join(output_folder, "adaptive_paragraph_bounding_boxes.png")
    cv2.imwrite(output_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"Paragraph bounding boxes saved to: {output_path}")

if __name__ == "__main__":
    input_image_path = "/path/to/your/input_image.png"
    output_folder = "/path/to/output_folder"
    detect_paragraphs(input_image_path, output_folder)
