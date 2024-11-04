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
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
    return img

def get_column_boundaries_from_words(prediction_groups, min_gap_width=50):
    x_centers = [(box[0][0] + box[1][0]) / 2 for word, box in prediction_groups[0]]
    x_centers = sorted(x_centers)
    column_boundaries = []
    start = x_centers[0]

    for i in range(1, len(x_centers)):
        if x_centers[i] - x_centers[i - 1] > min_gap_width:
            end = x_centers[i - 1]
            column_boundaries.append((start, end))
            start = x_centers[i]
    column_boundaries.append((start, x_centers[-1]))
    return column_boundaries

def extract_words_by_column(prediction_groups, column_boundaries):
    columns = [[] for _ in range(len(column_boundaries))]
    for word, box in prediction_groups[0]:
        x_center = (box[0][0] + box[1][0]) / 2
        for i, (start, end) in enumerate(column_boundaries):
            if start <= x_center < end:
                columns[i].append((word, box))
                break
    return columns

def cluster_paragraphs(column_words, eps=25, min_samples=1):
    paragraphs = []
    for words in column_words:
        if not words:
            continue
        word_centers = np.array([[(box[0][0] + box[1][0]) / 2, (box[0][1] + box[2][1]) / 2] for word, box in words])
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(word_centers)

        paragraph_dict = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                continue
            if label not in paragraph_dict:
                paragraph_dict[label] = []
            paragraph_dict[label].append(words[idx][1])

        paragraphs.extend(paragraph_dict.values())
    return paragraphs

def post_process_overlaps(paragraphs, overlap_threshold=10):
    adjusted_paragraphs = []
    for i, para1 in enumerate(paragraphs):
        x1_min, x1_max = min([box[0][0] for box in para1]), max([box[1][0] for box in para1])
        y1_min, y1_max = min([box[0][1] for box in para1]), max([box[2][1] for box in para1])
        overlap_found = False
        for j, para2 in enumerate(paragraphs):
            if i == j:
                continue
            x2_min, x2_max = min([box[0][0] for box in para2]), max([box[1][0] for box in para2])
            y2_min, y2_max = min([box[0][1] for box in para2]), max([box[2][1] for box in para2])

            if (x1_min < x2_max + overlap_threshold and x1_max > x2_min - overlap_threshold and
                y1_min < y2_max + overlap_threshold and y1_max > y2_min - overlap_threshold):
                overlap_found = True
                break

        if not overlap_found:
            adjusted_paragraphs.append(para1)
    return adjusted_paragraphs

def draw_bounding_boxes(image, paragraphs):
    for paragraph in paragraphs:
        min_x = min(box[0][0] for box in paragraph)
        max_x = max(box[1][0] for box in paragraph)
        min_y = min(box[0][1] for box in paragraph)
        max_y = max(box[2][1] for box in paragraph)
        cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
    return image

def paragraph_detection(input_image, output_folder):
    pipeline = keras_ocr.pipeline.Pipeline()
    img = preprocess_image(input_image)
    prediction_groups = pipeline.recognize([img])

    column_boundaries = get_column_boundaries_from_words(prediction_groups)
    column_words = extract_words_by_column(prediction_groups, column_boundaries)
    paragraphs = cluster_paragraphs(column_words)

    # Post-process to remove overlaps
    paragraphs = post_process_overlaps(paragraphs)
    img_with_boxes = draw_bounding_boxes(img.copy(), paragraphs)

    output_image_path = f"{output_folder}/bounding_boxes.png"
    cv2.imwrite(output_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

    print("Paragraph bounding boxes saved to:", output_image_path)

if __name__ == "__main__":
    input_image_path = "/mnt/data/17307220852961967422134678953908.jpg"
    output_folder = "/mnt/data/output"
    os.makedirs(output_folder, exist_ok=True)
    paragraph_detection(input_image_path, output_folder)
