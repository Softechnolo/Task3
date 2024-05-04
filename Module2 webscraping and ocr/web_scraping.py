from PIL import Image
import pytesseract
import os
import requests
from bs4 import BeautifulSoup

# Set the tesseract path in the script before calling image_to_string
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_core(filename):
    """
    This function will handle the OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

def download_images(base_url, start_page, end_page, path):
    """
    This function will download and save images from the given URL into the specified path.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for page_number in range(start_page, end_page + 1):
        url = f"{base_url}?page={page_number}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')

        for img in img_tags:
            img_url = img['src']
            if img_url.startswith('http'):
                response = requests.get(img_url, stream=True)
                with open(os.path.join(path, img_url.split("/")[-1]), 'wb') as out_file:
                    out_file.write(response.content)

def process_query(image_file):
    """
    This function will process the OCR-based query.
    """
    # Extract text from the image file
    text = ocr_core(image_file)

    # Process the extracted text (you'll need to implement this part based on the logic of Endpoint 1)
    processed_text = process_text(text)  # Assuming process_text is a function you have that processes the text

    return processed_text

# Call the functions
download_images("https://kanemtrade.com/search", 1, 65, "scrapped_images")
