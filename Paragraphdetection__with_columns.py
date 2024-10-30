import layoutparser as lp
import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def detect_layout(img):
    # Use a LayoutParser pre-trained model for document layout detection
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    layout = model.detect(img)
    
    # Filter only "Text" regions for column and paragraph segmentation
    text_blocks = [block for block in layout if block.type == "Text"]
    return text_blocks

def extract_text_from_column(img, column_box):
    x_1, y_1, x_2, y_2 = map(int, column_box.coordinates)
    column_img = img[y_1:y_2, x_1:x_2]
    d = pytesseract.image_to_data(column_img, output_type=Output.DICT, config='--psm 6')

    paragraphs = []
    current_paragraph = []
    for i in range(len(d['level'])):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        
        # Group lines into paragraphs by detecting vertical space
        if current_paragraph and abs(current_paragraph[-1][1] - y) > 20:  # 20 pixels as a dynamic example
            paragraphs.append(current_paragraph)
            current_paragraph = []
        
        current_paragraph.append((x, y, w, h, d['text'][i]))
    
    # Add the last paragraph
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def draw_paragraph_boxes(img, column_box, paragraphs):
    x_1, y_1, x_2, y_2 = map(int, column_box.coordinates)
    for paragraph in paragraphs:
        min_x = min([x + x_1 for x, _, w, _, _ in paragraph])
        max_x = max([x + w + x_1 for x, _, w, _, _ in paragraph])
        min_y = min([y + y_1 for _, y, _, _, _ in paragraph])
        max_y = max([y + h + y_1 for _, y, h, _, _ in paragraph])
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

def main(img_path):
    img = preprocess_image(img_path)
    
    # Step 1: Detect layout and extract text blocks (columns)
    text_blocks = detect_layout(img)
    
    # Step 2: Process each text block separately
    for text_block in text_blocks:
        paragraphs = extract_text_from_column(img, text_block)
        
        # Step 3: Draw bounding boxes for paragraphs
        draw_paragraph_boxes(img, text_block, paragraphs)
    
    # Save or display the result
    output_path = "paragraph_bounding_boxes.png"
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    img_path = 'your_image_path_here.jpg'
    main(img_path)
