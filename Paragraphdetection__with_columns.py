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

def detect_words(pipeline, img):
    prediction_groups = pipeline.recognize([img])
    return prediction_groups[0]  # Return predictions for the first image

def organize_into_columns(predictions, min_gap_width=50):
    x_centers = [(box[0][0] + box[1][0]) / 2 for _, box in predictions]
    sorted_x_centers = sorted(set(x_centers))
    column_boundaries = []
    start = sorted_x_centers[0]

    for i in range(1, len(sorted_x_centers)):
        if sorted_x_centers[i] - sorted_x_centers[i - 1] > min_gap_width:
            end = sorted_x_centers[i - 1]
            column_boundaries.append((start, end))
            start = sorted_x_centers[i]
    column_boundaries.append((start, sorted_x_centers[-1]))
    
    # Assign each word to a column based on x-center
    columns = [[] for _ in range(len(column_boundaries))]
    for word, box in predictions:
        x_center = (box[0][0] + box[1][0]) / 2
        for i, (start, end) in enumerate(column_boundaries):
            if start <= x_center < end:
                columns[i].append((word, box))
                break
    return columns

def group_into_paragraphs(words, eps=25):
    if not words:
        return []

    # Calculate the vertical center of each word
    word_centers = np.array([[(box[0][1] + box[2][1]) / 2] for _, box in words])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(word_centers)
    
    # Organize words into paragraphs based on cluster labels
    paragraphs = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in paragraphs:
            paragraphs[label] = []
        paragraphs[label].append(words[idx][1])  # Only store box for drawing

    # Sort paragraphs by y-coordinates to maintain reading order
    sorted_paragraphs = [sorted(para, key=lambda box: box[0][1]) for para in paragraphs.values()]
    return sorted_paragraphs

def draw_bounding_boxes(img, columns):
    for col_idx, column in enumerate(columns):
        for para_idx, paragraph in enumerate(column):
            min_x = min(box[0][0] for box in paragraph)
            max_x = max(box[1][0] for box in paragraph)
            min_y = min(box[0][1] for box in paragraph)
            max_y = max(box[2][1] for box in paragraph)
            cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
    return img

def paragraph_detection(input_image, output_folder):
    pipeline = keras_ocr.pipeline.Pipeline()
    img = preprocess_image(input_image)
    predictions = detect_words(pipeline, img)

    columns = organize_into_columns(predictions)
    column_paragraphs = [group_into_paragraphs(col) for col in columns]

    img_with_boxes = draw_bounding_boxes(img.copy(), column_paragraphs)
    
    output_image_path = f"{output_folder}/paragraph_boxes.png"
    cv2.imwrite(output_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print("Paragraph bounding boxes saved to:", output_image_path)

if __name__ == "__main__":
    input_image_path = "/mnt/data/17307224973138071674097826434452.jpg"
    output_folder = "/mnt/data/output"
    os.makedirs(output_folder, exist_ok=True)
    paragraph_detection(input_image_path, output_folder)
