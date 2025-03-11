import requests
from PIL import Image
import os

url = "http://localhost:5000/generate-layout"
data = {
    "room_width": 10,
    "room_height": 8,
    "furniture": [
           {"name": "Bed", "width": 2, "height": 3},
    {"name": "Table", "width": 2, "height": 2},
    {"name": "Sofa", "width": 3, "height": 2},
    {"name": "Chair", "width": 1, "height": 1},
    {"name": "Wardrobe", "width": 2, "height": 2},
   
    ]
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an exception for bad status codes
    result = response.json()
    print("Response:", result)
    # Display the image if it exists
    if "image_path" in result:
        image_path = result["image_path"]
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img.show()  # Opens the image in the default viewer
        else:
            print(f"Image not found at {image_path}")
    else:
        print("No image path in response.")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")