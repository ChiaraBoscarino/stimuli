from PIL import Image
import numpy as np


# Assume 'npy_stack_frames' is a list of NumPy arrays or file paths representing individual frames
# Example: npy_stack_frames = [image1, image2, image3, ...]

def create_gif(npy_stack_frames, output_path, dt=100, loop=0):
    """
    Generate a GIF from a list of npy_stack_frames.

    Args:
    npy_stack_frames (list): List of image file paths or numpy arrays.
    output_path (str): The output path for the GIF.
    dt (int): Duration of each frame in milliseconds.
    loop (int): Number of times the GIF will loop. 0 for infinite loop.
    """
    # Convert from binary to 0-255 if necessary
    if np.max(npy_stack_frames) == 1:
        npy_stack_frames = npy_stack_frames*255

    # Convert NumPy arrays to PIL npy_stack_frames if necessary
    pil_images = []
    for img in npy_stack_frames:
        if isinstance(img, np.ndarray):
            pil_images.append(Image.fromarray(img))
        else:
            pil_images.append(Image.open(img))

    # Save the npy_stack_frames as a GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=dt,
        loop=loop
    )


# # Example usage
# # Assume you have a list of numpy npy_stack_frames or file paths in 'npy_stack_frames'
# images = [np.random.randint(0, 255, (100, 100), dtype=np.uint8) for _ in range(10)]  # Example stack of 10 random npy_stack_frames
# create_gif(images, 'output.gif', duration=200)
