import numpy as np
import layoutparser as lp
import torch
import cv2

# Update this path to your image
image_path = '/home/azureuser/lekhaanuvaad_processing/Test_images/Gazette_Page_01.jpg'  
img = cv2.imread(image_path)



# Load model1 with optimized score threshold
model1 = lp.Detectron2LayoutModel(
    config_path="/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/config1.yml",
    model_path="/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/model1.pth",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.15],  # Lowered threshold for better detection
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Define color mapping for different types
color_map = {
    "Text": (255, 0, 0),    # Red
    #"Title": (0, 255, 0),   # Green
    #"List": (0, 0, 255),    # Blue
    "Table": (255, 255, 0), # Cyan
    #"Figure": (255, 0, 255) # Magenta
}

def draw_boxes(image, layout):
    """Draw bounding boxes on the image based on the layout."""
    for block in layout:
        block_type = block.type
        if block_type in color_map:  # Check if the block type is in our color map
            x1, y1, x2, y2 = block.coordinates
            color = color_map[block_type]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw box
            # Add text label
            cv2.putText(image, block_type, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the image with drawn boxes
    output_image_path = "output_image_model1.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved as: {output_image_path}")

# Load the image and detect layout
layout_result1 = model1.detect(img)

# Draw boxes and save the image
draw_boxes(img.copy(), layout_result1)

# Perform OCR on detected text blocks
ocr_agent = lp.TesseractAgent(languages='eng')

for block in layout_result1:
    # Crop image around the detected layout
    segment_image = (block
                     .pad(left=15, right=15, top=5, bottom=5)
                     .crop_image(img))
    
    # Perform OCR
    text = ocr_agent.detect(segment_image)

    # Save OCR result
    block.set(text=text, inplace=True)

# Print the detected text
# for txt in layout_result1:
#     print("Text = ", txt.text)
#     print("Coordinates = ", txt.coordinates)
#     print("---")
