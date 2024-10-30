import numpy as np
import layoutparser as lp
import cv2

# Update this path to your image
image_path = '/home/azureuser/lekhaanuvaad_processing/Test_images/Gazette_Page_01.jpg'  
img = cv2.imread(image_path)

# Load the Detectron2 layout model
model = lp.Detectron2LayoutModel(
    config_path="/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/config1.yml",
    model_path="/home/azureuser/lekhaanuvaad_processing/paragraph_detection/pretrained_model/model1.pth",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],  # Adjust threshold
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Define color mapping for different types
color_map = {
    "Text": (255, 0, 0),  # Red for text blocks
    "Title": (0, 255, 0),  # Green for titles (if needed)
    "List": (0, 0, 255),  # Blue for lists (if needed)
    "Table": (255, 255, 0),  # Cyan for tables
    "Figure": (255, 0, 255)  # Magenta for figures
}

def draw_boxes(image, layout):
    """Draw bounding boxes on the image based on the layout."""
    for block in layout:
        if block.type in color_map:  # Check if the block type is in our color map
            x1, y1, x2, y2 = block.coordinates
            color = color_map[block.type]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw box
            # Add text label
            cv2.putText(image, block.type, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the image with drawn boxes
    output_image_path = "output_image_detectron.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved as: {output_image_path}")

# Load the image and detect layout
layout_result = model.detect(img)

# Draw boxes on the original image
draw_boxes(img.copy(), layout_result)

# Function to merge nearby boxes to group text into paragraphs or columns
def merge_boxes(layout, merge_threshold=15):
    merged_boxes = []
    for block in layout:
        x_min, y_min, x_max, y_max = block.coordinates
        merged = False
        
        # Check if this box should be merged with an existing box
        for mbox in merged_boxes:
            if (x_min < mbox[2] + merge_threshold and x_max > mbox[0] - merge_threshold and
                y_min < mbox[3] + merge_threshold and y_max > mbox[1] - merge_threshold):
                # Merge boxes
                mbox[0] = min(mbox[0], x_min)
                mbox[1] = min(mbox[1], y_min)
                mbox[2] = max(mbox[2], x_max)
                mbox[3] = max(mbox[3], y_max)
                merged = True
                break
        if not merged:
            merged_boxes.append([x_min, y_min, x_max, y_max])
    
    return merged_boxes

# Merge boxes for paragraph-level grouping
merged_boxes = merge_boxes(layout_result)

# Draw merged boxes on image
for (x_min, y_min, x_max, y_max) in merged_boxes:
    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)  # Red for merged boxes

# Save and display the output image with paragraph and column boundaries
output_image_path = "output_image_paragraphs.jpg"
cv2.imwrite(output_image_path, img)
print(f"Output image saved as: {output_image_path}")

# Clean up the model resources
del model
