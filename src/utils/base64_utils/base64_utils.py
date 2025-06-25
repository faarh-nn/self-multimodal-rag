import base64
import io
import os
import re
from IPython.display import HTML, display
from PIL import Image


def encode_image_from_bytes(image_bytes):
    '''
    Encode image bytes as base64
    '''
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_image_to_bytes(b64_string):
    '''
    Decode base64 string to image bytes
    '''
    return base64.b64decode(b64_string.encode("utf-8"))


def encode_image_from_path(image_path):
    '''
    Encode image from file path as base64
    '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
    
def load_image(inputs: dict) -> dict:
    '''
    Load image from file or image bytes and encode it as base64
    '''
    image = inputs["image"]
    if os.path.exists(image):   # image is a file path
        image_base64 = encode_image_from_path(image)
    else:   # image is provided as bytes
        image_base64 = encode_image_from_bytes(image)
    return {"image": image_base64}


def looks_like_base64(sb):
    '''
    Check if the string looks like base64
    '''
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    '''
    Check if the base64 data is an image by looking at the start of the data
    '''
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    '''
    Resize an image encoded as a Base64 string
    '''
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def plt_img_base64(img_base64):
    '''
    Display base64 encoded string as image
    '''
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/png;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))