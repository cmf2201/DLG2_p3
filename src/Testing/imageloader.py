import cv2
from PIL import Image

# Load an image using OpenCV
image = cv2.imread('/home/ctnguyen/neural_nemesis/DLG2_p3/src/dataset/2011_09_28/2011_09_28_drive_0002_sync/image_01/data/0000000002.png')

# Convert the image from BGR (OpenCV format) to RGB (PIL format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the NumPy array to a PIL image
pil_image = Image.fromarray(image_rgb)

# Save the PIL image to a file
pil_image.save('output_image.jpg')  # You can change the file format and name as needed