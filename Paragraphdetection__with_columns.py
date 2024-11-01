import cv2
import numpy as np
import layoutparser as lp
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Load the PubLayNet model from LayoutParser
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config", 
    model_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3],  # Adjusted threshold
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Initialize the CRAFT text detector
craft_detector = lp.models.CRAFT()

def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    processed_image = cv2.adaptiveThreshold(blurred_image, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
    return processed_image

def detect_layout(image):
    layout = model.detect(image)
    print(f"Detected layout blocks: {layout}")  # Inspect detected layout
    return layout

def merge_close_blocks(layout, distance_threshold=10):
    merged_blocks = []
    for block in layout:
        if not merged_blocks:
            merged_blocks.append(block)
        else:
            last_block = merged_blocks[-1]
            if abs(last_block.coordinates[1] - block.coordinates[1]) < distance_threshold:
                # Merge blocks logic here (simple average for merging)
                last_block.coordinates = [
                    min(last_block.coordinates[0], block.coordinates[0]),
                    min(last_block.coordinates[1], block.coordinates[1]),
                    max(last_block.coordinates[2], block.coordinates[2]),
                    max(last_block.coordinates[3], block.coordinates[3]),
                ]
            else:
                merged_blocks.append(block)
    return merged_blocks

def process_layout(image, layout):
    for block in layout:
        if block.type == "Text":  # Match the correct type
            print(f"Detected block: {block}")  # Debug print
            x_1, y_1, x_2, y_2 = map(int, block.coordinates)
            cv2.rectangle(image, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
    return image

def detect_textlines(image):
    # Detect text lines using CRAFT
    text_lines = craft_detector.detect(image)
    return text_lines

def ocr_textlines(image, text_lines):
    ocr_model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
    results = []
    for line in text_lines:
        x_1, y_1, x_2, y_2 = map(int, line.coordinates)
        text_region = image[y_1:y_2, x_1:x_2]
        doc = DocumentFile.from_images(text_region)
        result = ocr_model(doc)
        results.append(result)
    return results

def save_output(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Output saved at {output_path}")

if __name__ == "__main__":
    img_path = '/mnt/data/17304896446951666356884000969127.jpg'  # Input image path
    output_path = '/mnt/data/processed_output.jpg'

    # Load and preprocess image
    image = load_image(img_path)
    preprocessed_image = preprocess_image(image)

    # Detect layout
    layout = detect_layout(preprocessed_image)
    
    # Optionally merge close blocks
    merged_layout = merge_close_blocks(layout)

    # Process and annotate layout blocks
    processed_image = process_layout(image.copy(), merged_layout)

    # Detect text lines using CRAFT
    text_lines = detect_textlines(preprocessed_image)

    # Perform OCR on detected text lines
    ocr_results = ocr_textlines(preprocessed_image, text_lines)

    # Save the output with detected layout blocks
    save_output(processed_image, output_path)

    # Print OCR results for each detected text line
    for i, result in enumerate(ocr_results):
        print(f"Text line {i}: {result}")
