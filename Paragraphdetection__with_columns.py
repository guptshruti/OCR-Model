import numpy as np
import layoutparser as lp
import torch
import cv2
from sklearn.cluster import DBSCAN

# Load your image
image_path = '/home/azureuser/lekhaanuvaad_processing/Test_images/Gazette_Page_01.jpg'
img = cv2.imread(image_path)

# Model Configuration and Threshold Tuning
model1 = lp.Detectron2LayoutModel(
    config_path="/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/config1.yml",
    model_path="/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/model1.pth",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.15],  # Adjusted threshold
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Define color mapping for different types
color_map = {
    "Text": (255, 0, 0),    # Red
    "Table": (255, 255, 0), # Cyan
}

def preprocess_image(image):
    """Enhance image contrast and apply denoising to improve layout detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 30, 7, 21)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def draw_boxes(image, layout):
    """Draw bounding boxes on the image based on the layout."""
    for block in layout:
        block_type = block.type
        if block_type in color_map:
            x1, y1, x2, y2 = block.coordinates
            color = color_map[block_type]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, block_type, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the image with drawn boxes
    output_image_path = "output_image_model1_enhanced.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved as: {output_image_path}")

def group_text_blocks(layout, eps=50):
    """Group nearby text boxes to improve paragraph/column separation."""
    coords = np.array([block.coordinates[:2] for block in layout if block.type == "Text"])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)

    grouped_layout = []
    for idx, label in enumerate(clustering.labels_):
        grouped_layout.append((label, layout[idx]))

    return grouped_layout

# Preprocess image
preprocessed_img = preprocess_image(img)

# Load the image and detect layout
layout_result1 = model1.detect(preprocessed_img)

# Grouping text boxes by proximity to improve structure
grouped_layout = group_text_blocks(layout_result1)

# Draw boxes on original image for better clarity
draw_boxes(img.copy(), layout_result1)

# Perform OCR on detected text blocks
ocr_agent = lp.TesseractAgent(languages='eng')

for label, block in grouped_layout:
    segment_image = (block
                     .pad(left=15, right=15, top=5, bottom=5)
                     .crop_image(img))

    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)

# Print the detected text
for txt in layout_result1:
    print("Text:", txt.text)
    print("Coordinates:", txt.coordinates)
    print("---")
