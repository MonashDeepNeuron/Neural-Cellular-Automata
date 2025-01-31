"""

"""

import os
from PIL import Image
import numpy as np

def adjust_trimap_colors(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the new colors for the trimap
    colors = {
        0: [0, 0, 255],    # Background: Red
        1: [0, 255, 0],    # Object: Green
        2: [255, 0, 0]     # Boundary: Blue
    }

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Load the image
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path).convert("L")  # Convert to grayscale

            # Convert the image to a NumPy array
            img_array = np.array(image)

            # Create a new array for the colored output
            colored_output = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)

            # Apply the new colors to the trimap
            for cls, color in colors.items():
                colored_output[img_array == cls] = color

            # Convert the colored output back to an image
            colored_image = Image.fromarray(colored_output)

            # Save the colored image to the output folder
            output_path = os.path.join(output_folder, filename)
            colored_image.save(output_path)

            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_folder = 'C:/Neural-Cellular-Automata/ImageSegmentation/trimaps'
    output_folder = 'C:/Neural-Cellular-Automata/ImageSegmentation/trimaps_colored'
    adjust_trimap_colors(input_folder, output_folder)