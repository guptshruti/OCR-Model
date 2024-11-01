import cv2
import numpy as np
import layoutparser as lp
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Load the PubLayNet model from LayoutParser
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config", 
    model_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
)

def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def detect_layout(image):
    # Detect layout blocks (text, title, etc.)
    layout = model.detect(image)
    return layout

def process_layout(image, layout):
    for block in layout:
        if block.type == "text":
            x_1, y_1, x_2, y_2 = map(int, block.coordinates)
            text_region = image[y_1:y_2, x_1:x_2]
            cv2.rectangle(image, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
    return image

def detect_textlines(text_region):
    # Initialize the CRAFT text detector
    craft_detector = lp.models.CRAFT()
    text_lines = craft_detector.detect(text_region)
    return text_lines

def ocr_textlines(text_lines, image):
    # Perform OCR on detected text lines using Doctr
    doc = DocumentFile.from_images(image)
    ocr_model = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
    result = ocr_model(doc)
    return result

def save_output(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Output saved at {output_path}")

if __name__ == "__main__":
    img_path = '/mnt/data/17304896446951666356884000969127.jpg'  # Input image path
    output_path = '/mnt/data/processed_output.jpg'

    # Load image and perform layout detection
    image = load_image(img_path)
    layout = detect_layout(image)

    # Process and annotate layout blocks
    processed_image = process_layout(image, layout)

    # Save the output with detected layout blocks
    save_output(processed_image, output_path)
