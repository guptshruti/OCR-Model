import layoutparser as lp
import cv2
import numpy as np

# Load the image
image_path = "your_image.jpg"  # Update with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for processing

# Load the pre-trained PubLayNet model
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Detect layout
layout = model.detect(image_rgb)

# Filter for text blocks
text_blocks = lp.Layout([b for b in layout if b.type == "Text"])

# Convert bounding boxes to a usable format
boxes = [block.coordinates for block in text_blocks]

# Sort by X coordinate to group columns
boxes = sorted(boxes, key=lambda x: x[0])

# Separate into columns based on X-spacing threshold (e.g., 50 pixels)
columns = []
current_column = [boxes[0]]

for i in range(1, len(boxes)):
    if abs(boxes[i][0] - boxes[i - 1][0]) < 50:  # Adjust threshold as needed
        current_column.append(boxes[i])
    else:
        columns.append(current_column)
        current_column = [boxes[i]]
columns.append(current_column)

# Draw bounding boxes for each column and individual text blocks
for column in columns:
    for box in column:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

# Save the output image
output_image_path = "output_image_with_boxes.jpg"
cv2.imwrite(output_image_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

print(f"Output image saved as: {output_image_path}")
