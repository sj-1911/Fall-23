from IPython.display import Image, display
from PIL import Image as PILImage
import requests
from io import BytesIO

#Pull image from URL and be able to convert it
image_url = "https://images.boats.com/resize/1/35/59/8463559_20220825085347526_1_LARGE.jpg?t=1661443454000"
response = requests.get(image_url)
img = PILImage.open(BytesIO(response.content))

# Convert the image to grayscale
gray_img = img.convert('L')
gray_img_path = 'grayscale_image.jpg'
gray_img.save(gray_img_path)
display(Image(filename=gray_img_path))

from IPython.display import Image, display
from PIL import Image as PILImage
import requests
from io import BytesIO

# Define the URL of the image
image_url = "https://images.boats.com/resize/1/35/59/8463559_20220825085347526_1_LARGE.jpg?t=1661443454000"

# Fetch the image from the URL
response = requests.get(image_url)
img = PILImage.open(BytesIO(response.content))

# Display the grayscale image
display(Image(data=img.tobytes()))

# Get and display the shape (dimensions) of the original image
original_image_shape = img.size
print("Original Image Shape:", original_image_shape)

# Add image from URL
desired_width = 224
desired_height = 224

# Fetch the image from the URL
response = requests.get(image_url)
img = PILImage.open(BytesIO(response.content))

# Resize the image
img = img.resize((desired_width, desired_height))

# Display the resized image
display(img)
#Resize image to 200x300
grayscale_img = img.convert("L")

# Display the grayscale image
display(grayscale_img)
#Greyscale Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display an image
def display_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Convert the grayscale image to a numpy array
grayscale_array = np.array(grayscale_img)

# Number of random filters
num_filters = 10

# Size of each filter (you can adjust this)
filter_size = 5

# Generate random filters
filters = [np.random.randn(filter_size, filter_size) for _ in range(num_filters)]

# Display the random filters
for i, filter in enumerate(filters):
    display_image(filter, title=f"Filter {i+1}")

# Convolve the image with the filters and display feature maps
feature_maps = [cv2.filter2D(grayscale_array, -1, filter) for filter in filters]

# Display the feature maps
for i, feature_map in enumerate(feature_maps):
    display_image(feature_map, title=f"Feature Map {i+1}")
