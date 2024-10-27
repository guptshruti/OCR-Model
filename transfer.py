
from PIL import Image, ImageDraw, ImageFont
import json
import re

def load_coordinates(file_path):
    # Load paragraph coordinates from the text file
    with open(file_path, 'r') as f:
        raw_data = f.read().strip().split('\n\n')
    coordinates = []
    for block in raw_data:
        matches = re.findall(r'\((\d+),\s*(\d+)\)', block)
        if matches:
            start = tuple(map(int, matches[0]))
            end = tuple(map(int, matches[1]))
            coordinates.append((start, end))
    return coordinates

def load_text(file_path):
    # Load extracted text from JSON file
    with open(file_path, 'r') as f:
        return json.load(f)

def draw_text_in_box(draw, text, box, font_path="arial.ttf", max_font_size=100, min_font_size=10):
    x0, y0 = box[0]
    x1, y1 = box[1]
    box_width = x1 - x0
    box_height = y1 - y0

    # Remove newline characters for fitting text within the box
    text = text.replace('\n', ' ')
    
    # Binary search to fit text within the box
    font_size = max_font_size
    while font_size >= min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
        
        # Check if the text fits within the box dimensions
        if text_width <= box_width and text_height <= box_height:
            break
        font_size -= 1  # Reduce font size until text fits

    # Center text within the box
    text_x = x0 + (box_width - text_width) / 2
    text_y = y0 + (box_height - text_height) / 2
    draw.text((text_x, text_y), text, font=font, fill="black")

def main(image_path, coordinates_path, text_path, output_path):
    # Load base image
    base_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base_image)

    # Load coordinates and extracted text
    coordinates = load_coordinates(coordinates_path)
    extracted_text = load_text(text_path)

    # Iterate over each paragraph and draw the text within its boundaries
    for i, (start, end) in enumerate(coordinates):
        paragraph_key = f"paragraph_{i+1}.png"
        if paragraph_key in extracted_text:
            paragraph_text = extracted_text[paragraph_key]
            if paragraph_text:
                draw_text_in_box(draw, paragraph_text, (start, end))

    # Save the final image
    base_image.save(output_path)
    print(f"Output saved at: {output_path}")

if __name__ == "__main__":
    main(
        image_path="base_image.png",
        coordinates_path="coordinates.txt",
        text_path="text.json",
        output_path="output_image.png"
    )
