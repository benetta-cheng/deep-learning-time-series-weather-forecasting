from PIL import Image

def load_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Display it (in Jupyter or similar)
    img.show()

    # Convert to NumPy array if needed
    import numpy as np
    img_array = np.array(img)