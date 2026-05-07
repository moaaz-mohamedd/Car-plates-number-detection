import cv2
import matplotlib.pyplot as plt
import numpy as np


def bgr_to_rgb(image):
    """
    Convert image from BGR to RGB.

    Why?
    OpenCV reads images in BGR format.
    Matplotlib and Streamlit display images in RGB format.
    """
    if image is None:
        return None

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image(title, image, cmap=None, figsize=(10, 6)):
    """
    Display one image in the notebook.
    """
    plt.figure(figsize=figsize)
    plt.title(title)

    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap or "gray")
    else:
        plt.imshow(bgr_to_rgb(image))

    plt.axis("off")
    plt.show()


def show_images_grid(images_dict, cols=2, figsize=(14, 10)):
    """
    Display multiple images in a grid.

    images_dict example:
    {
        "Original": original_image,
        "Gray": gray_image
    }
    """
    names = list(images_dict.keys())
    images = list(images_dict.values())

    rows = int(np.ceil(len(images) / cols))

    plt.figure(figsize=figsize)

    for i, (name, image) in enumerate(zip(names, images)):
        plt.subplot(rows, cols, i + 1)
        plt.title(name)

        if len(image.shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(bgr_to_rgb(image))

        plt.axis("off")

    plt.tight_layout()
    plt.show()


def crop_with_padding(image, box, padding=8):
    """
    Crop a region from image with small padding around it.

    box format:
    (x, y, w, h)
    """
    x, y, w, h = box
    H, W = image.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    return image[y1:y2, x1:x2]