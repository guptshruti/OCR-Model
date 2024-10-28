import json
from langdetect import detect
from PIL import ImageFont

# Path to Indic fonts
font_map = {
    'hi': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansDevanagari-Regular.ttf',  # Hindi
    'bn': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansBengali-Regular.ttf',  # Bengali
    'gu': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansGujarati-Regular.ttf',  # Gujarati
    'kn': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansKannada-Regular.ttf',  # Kannada
    'ml': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansMalayalam-Regular.ttf',  # Malayalam
    'or': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansOriya-Regular.ttf',  # Oriya
    'pa': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansGurmukhi-Regular.ttf',  # Punjabi
    'ta': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansTamil-Regular.ttf',  # Tamil
    'te': '/usr/share/fonts/truetype/indic/noto_fonts/noto-fonts-main/hinted/ttf/NotoSansTelugu-Regular.ttf',  # Telugu
    # Add other languages as needed
}

def get_font_for_text(paragraph_text, font_size=20):
    """Detect language of paragraph text and return corresponding font."""
    try:
        # Detect language code (e.g., 'hi' for Hindi, 'bn' for Bengali)
        lang_code = detect(paragraph_text)
        font_path = font_map.get(lang_code)
        
        if font_path:
            return ImageFont.truetype(font_path, font_size)
        else:
            print(f"Language '{lang_code}' not found in font_map. Using default font.")
            return ImageFont.load_default()  # Fallback to default font if language not found
    except Exception as e:
        print(f"Error detecting language or loading font: {e}")
        return ImageFont.load_default()

# Example function to process JSON file with extracted text
def process_text_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        extracted_text_data = json.load(f)
    
    for paragraph, text in extracted_text_data.items():
        font = get_font_for_text(text)
        print(f"Paragraph: {paragraph}\nLanguage Detected: {detect(text)}\nFont: {font.path}\n")

# Example usage
if __name__ == "__main__":
    # Path to your JSON file with extracted text
    json_file_path = 'path/to/extracted_text.json'
    process_text_json(json_file_path)
