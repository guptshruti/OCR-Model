import keras_ocr
import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def inpaint_paragraphs(img_path, pipeline):
    img = preprocess_image(img_path)
    prediction_groups = pipeline.recognize([img])
    
    # Visualize word-level bounding boxes
    for word, box in prediction_groups[0]:
        # Correctly format the points for the rectangle function
        top_left = tuple(map(int, box[0]))
        bottom_right = tuple(map(int, box[2]))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Draw green boxes around words

    paragraphs = []
    threshold_y = 20  # Vertical threshold to consider a new paragraph
    current_paragraph = []

    for word, box in prediction_groups[0]:
        if not current_paragraph:
            current_paragraph.append(box)
        else:
            last_word_y1 = current_paragraph[-1][2][1]  # y-coordinate of the last word's bottom
            if box[0][1] - last_word_y1 > threshold_y:
                paragraphs.append(current_paragraph)
                current_paragraph = [box]
            else:
                current_paragraph.append(box)

    if current_paragraph:
        paragraphs.append(current_paragraph)

    # Draw bounding boxes for paragraphs
    for paragraph in paragraphs:
        min_x = min(coords[0][0] for coords in paragraph)
        max_x = max(coords[1][0] for coords in paragraph)
        min_y = min(coords[0][1] for coords in paragraph)
        max_y = max(coords[2][1] for coords in paragraph)
        cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)

    return img, paragraphs

# Initialize the keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Example usage
img_path = '/home/azureuser/lekhaanuvaad_processing/Test_images/5.png'
img_paragraphs, paragraph_coords = inpaint_paragraphs(img_path, pipeline)

# Save the output image with bounding boxes
output_image_path = '/home/azureuser/lekhaanuvaad_processing/paragraph_detection/paragraph_bounding_boxes.png'
cv2.imwrite(output_image_path, cv2.cvtColor(img_paragraphs, cv2.COLOR_BGR2RGB))

# Print the paragraph coordinates
print("Paragraph bounding boxes coordinates:", paragraph_coords)
