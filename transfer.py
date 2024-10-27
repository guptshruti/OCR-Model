from PIL import Image, ImageDraw, ImageFont
import json

def draw_text_in_box(draw, text, box_coords, font, max_font_size=30, min_font_size=5):
    """
    Draw text within specified bounding box coordinates, resizing text as needed to fit.
    """
    start, end = box_coords
    box_width = end[0] - start[0]
    box_height = end[1] - start[1]
    font_size = max_font_size

    while font_size >= min_font_size:
        font = ImageFont.truetype(font.path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if text_width <= box_width and text_height <= box_height:
            text_x = start[0] + (box_width - text_width) // 2
            text_y = start[1] + (box_height - text_height) // 2
            draw.text((text_x, text_y), text, font=font, fill="black")
            return
        font_size -= 1

    # Draw text if it couldn't be resized to fit; it will likely overflow.
    text_x = start[0]
    text_y = start[1]
    draw.text((text_x, text_y), text, font=font, fill="black")

def draw_final(image_path, coordinates_path, text_path, output_path, font_path="arial.ttf"):
    """
    Draws text from a JSON file within bounding boxes on a given image.
    """
    # Load base image
    base_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(base_image)

    # Load coordinates and text data
    with open(coordinates_path, "r") as f:
        coordinates = [eval(line.strip()) for line in f if line.strip()]

    with open(text_path, "r") as f:
        text_data = json.load(f)

    # Initialize font
    font = ImageFont.truetype(font_path, size=20)  # Initial font size; will adjust

    for i, (start, end) in enumerate(coordinates):
        paragraph_key = f"paragraph_{i + 1}.png"
        paragraph_text = text_data.get(paragraph_key, "")

        if paragraph_text:
            draw_text_in_box(draw, paragraph_text, (start, end), font)

    # Save the final image
    base_image.save(output_path)
    print(f"Text drawn and saved to {output_path}")

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    coordinates_path = "path/to/coordinates.txt"
    text_path = "path/to/text.json"
    output_path = "path/to/output_image.jpg"
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Example path; adjust as needed

    draw_final(image_path, coordinates_path, text_path, output_path, font_path)
