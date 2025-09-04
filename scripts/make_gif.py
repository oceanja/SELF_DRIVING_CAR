import imageio
import os
from glob import glob

# Folder containing saved output frames
image_folder = "output_frames"
output_gif = "demo_output.gif"

# Get sorted list of image files
image_files = sorted(glob(os.path.join(image_folder, "*.jpg")))

# Load images
images = [imageio.imread(img) for img in image_files]

# Save GIF (duration in seconds per frame)
imageio.mimsave(output_gif, images, duration=0.05)

print(f"GIF saved to {output_gif}")