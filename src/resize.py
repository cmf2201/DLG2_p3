import os
from PIL import Image

def resize_images_in_directory(directory, n, m):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Create a folder for resized images
    resized_dir = os.path.join(directory, 'resized')
    os.makedirs(resized_dir, exist_ok=True)

    # Loop over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the file is an image (by extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Open an image file
                with Image.open(file_path) as img:
                    # Resize the image
                    resized_img = img.resize((n, m))

                    # Save the resized image in the new directory
                    resized_img.save(os.path.join(resized_dir, filename))

                    print(f"Resized and saved: {filename}")
            except Exception as e:
                print(f"Failed to resize {filename}: {e}")
        else:
            print(f"Skipping non-image file: {filename}")

if __name__ == "__main__":
    # Define the directory path and the size you want to resize to (n x m)
    directory_path = "/home/cmfrench/RBE474X/DLG2_p3/src/Before"  # Change this to your directory
    n, m = 640, 480  # Example size

    resize_images_in_directory(directory_path, n, m)
