import cv2


def resize_image(image, width=900):
    """
    Resize image while keeping the original aspect ratio.

    Why?
    Different images have different sizes.
    Resizing makes the detection logic more stable.
    """
    h, w = image.shape[:2]

    ratio = width / float(w)
    new_height = int(h * ratio)

    resized = cv2.resize(image, (width, new_height))

    return resized


def convert_to_grayscale(image):
    """
    Convert BGR image to grayscale.

    Why?
    License plate detection mainly depends on shapes, edges, and contrast.
    Grayscale simplifies processing because it uses one channel instead of three.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


def remove_noise(gray_image):
    """
    Reduce noise while preserving edges using Bilateral Filter.

    Why Bilateral Filter?
    It smooths the image but keeps important edges,
    which is useful for license plate boundaries. # gaussianBlur can blur edges, which may hurt detection performance.
    """
    denoised = cv2.bilateralFilter(gray_image, d=11, sigmaColor=17, sigmaSpace=17)

    return denoised


def enhance_contrast(gray_image):
    """
    Improve contrast using CLAHE.

    CLAHE = Contrast Limited Adaptive Histogram Equalization

    Why?
    It improves local contrast and makes plate details more visible,
    especially in poor lighting conditions or uneven illumination.its more better than histogram equalization
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray_image)

    return enhanced


def preprocess_image(image, width=900):
    """
    Full preprocessing pipeline.

    Input:
        Original BGR image

    Output:
        Dictionary containing all preprocessing stages
    """
    resized = resize_image(image, width=width)
    gray = convert_to_grayscale(resized)
    denoised = remove_noise(gray)
    enhanced = enhance_contrast(denoised)

    return {
        "resized": resized,
        "gray": gray,
        "denoised": denoised,
        "enhanced": enhanced
    }