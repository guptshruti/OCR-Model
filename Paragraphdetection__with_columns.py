import numpy as np
import layoutparser as lp
import torch
import cv2
from craft_text_detector import Craft

# Paths to your images and model configurations
image_path = '/home/azureuser/lekhaanuvaad_processing/Test_images/Gazette_Page_01.jpg'
model_paths = [
    "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/model1.pth",
    "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/model2.pth",
    "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/model3.pth",
]
config_paths = [
    "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/config1.yml",
    "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/config2.yml",
    "/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/config3.yml",
]

# Load image
img = cv2.imread(image_path)

# Initialize CRAFT for text region detection
craft = Craft(output_dir='./craft_output', crop_type="poly", cuda=True)

# Load all three models for ensemble
models = [lp.Detectron2LayoutModel(config, model_path=path, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.15], 
           label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}) for config, path in zip(config_paths, model_paths)]

def ensemble_detection(models, image):
    """Run the ensemble of models and aggregate results."""
    all_blocks = []
    for model in models:
        layout = model.detect(image)
        all_blocks.extend(layout)  # Aggregate detections from all models
    return all_blocks

def draw_boxes(image, layout, color_map):
    """Draw bounding boxes on the image based on the layout."""
    for block in layout:
        block_type = block.type
        if block_type in color_map:
            x1, y1, x2, y2 = block.coordinates
            color = color_map[block_type]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, block_type, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    output_image_path = "output_image_ensemble.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved as: {output_image_path}")

# Detect text-heavy regions with CRAFT
craft_result = craft.detect_text(image_path)
text_boxes = [lp.Rectangle(*box['box']) for box in craft_result["boxes"]]

# Ensemble detection
layout_result = ensemble_detection(models, img)

# Post-process with morphological operations
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
dilated = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert contours to layout parser format
contour_boxes = [lp.Rectangle(x, y, x+w, y+h) for cnt in contours for x, y, w, h in [cv2.boundingRect(cnt)]]

# Merge CRAFT and ensemble results
merged_boxes = text_boxes + layout_result + contour_boxes

# Draw combined results
draw_boxes(img.copy(), merged_boxes, color_map={"Text": (255, 0, 0), "Table": (255, 255, 0)})

# Cleanup CRAFT files after detection
craft.unload_craft_model()
