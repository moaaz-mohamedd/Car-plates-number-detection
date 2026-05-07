import cv2
import numpy as np

from src.preprocessing import remove_noise, enhance_contrast


def canny_pipeline(gray_image):
    """
    Canny-based detection pipeline.

    Steps:
    1. Noise removal
    2. Contrast enhancement
    3. Canny edge detection
    4. Morphological closing
    5. Dilation

    Output:
        Dictionary containing each processing step
    """

    # Step 1: reduce noise but keep important edges
    denoised = remove_noise(gray_image)

    # Step 2: improve local contrast
    enhanced = enhance_contrast(denoised)

    # Step 3: automatic Canny thresholds based on median intensity
    median_intensity = np.median(enhanced)

    lower_threshold = int(max(0, 0.66 * median_intensity))
    upper_threshold = int(min(255, 1.33 * median_intensity))

    edges = cv2.Canny(
        enhanced,
        lower_threshold,
        upper_threshold
    )

    # Step 4: use a rectangular kernel because plates are rectangular
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 5)
    )

    # Step 5: closing connects broken edges
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Step 6: dilation makes candidate regions stronger
    dilated = cv2.dilate(
        closed,
        kernel,
        iterations=1
    )

    return {
        "denoised": denoised,
        "enhanced": enhanced,
        "edges": edges,
        "closed": closed,
        "dilated": dilated,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold
    }


def blackhat_pipeline(gray_image):
    """
    Blackhat + Sobel pipeline.

    Why?
    Canny sometimes fails when the image has many edges.
    Blackhat helps highlight dark text-like details on bright regions,
    which is useful for license plates.

    Steps:
    1. Blackhat morphology
    2. Sobel X gradient
    3. Gaussian blur
    4. Morphological closing
    5. Otsu thresholding
    6. Erosion + dilation
    """

    # Rectangular kernel to match the horizontal shape of plates
    rect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 7)
    )

    # Blackhat highlights dark details on bright background
    blackhat = cv2.morphologyEx(
        gray_image,
        cv2.MORPH_BLACKHAT,
        rect_kernel
    )

    # Sobel X detects vertical changes/details
    grad_x = cv2.Sobel(
        blackhat,
        ddepth=cv2.CV_32F,
        dx=1,
        dy=0,
        ksize=-1
    )

    grad_x = np.absolute(grad_x)

    min_val = np.min(grad_x)
    max_val = np.max(grad_x)

    if max_val - min_val != 0:
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))

    grad_x = grad_x.astype("uint8")

    # Smooth small noisy details
    blurred = cv2.GaussianBlur(
        grad_x,
        (5, 5),
        0
    )

    # Connect close text-like regions
    closed = cv2.morphologyEx(
        blurred,
        cv2.MORPH_CLOSE,
        rect_kernel
    )

    # Convert to binary image using automatic threshold
    thresh = cv2.threshold(
        closed,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    # Remove small noise
    eroded = cv2.erode(
        thresh,
        small_kernel,
        iterations=1
    )

    # Restore important regions after erosion
    dilated = cv2.dilate(
        eroded,
        small_kernel,
        iterations=2
    )

    return {
        "blackhat": blackhat,
        "sobel_x": grad_x,
        "blurred": blurred,
        "closed": closed,
        "threshold": thresh,
        "eroded": eroded,
        "dilated": dilated
    }
    
def extract_candidates(binary_image, original_image):
    """
    Extract possible license plate regions from a binary image.

    Simple version:
    - Find contours
    - Build bounding boxes
    - Keep only plate-like rectangles
    """

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = original_image.shape[:2]
    image_area = H * W

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w == 0 or h == 0:
            continue

        contour_area = cv2.contourArea(contour)
        rect_area = w * h

        aspect_ratio = w / float(h)
        area_ratio = rect_area / float(image_area)
        extent = contour_area / float(rect_area)
        center_y = (y + h / 2) / H

        if not (2.0 <= aspect_ratio <= 6.8):
            continue

        if not (0.0015 <= area_ratio <= 0.07):
            continue

        if w < 70 or h < 18:
            continue

        if extent < 0.12:
            continue

        # مش شرط اللوحة في النص، بس غالبًا مش فوق خالص
        if not (0.20 <= center_y <= 0.95):
            continue

        candidate = {
            "box": (x, y, w, h),
            "aspect_ratio": aspect_ratio,
            "area_ratio": area_ratio,
            "contour_area": contour_area,
            "extent": extent,
            "center_y": center_y
        }

        candidates.append(candidate)

    return candidates

def draw_candidates(image, candidates, top_n=None):
    """
    Draw candidate bounding boxes on the image.

    Green boxes represent candidate regions.
    """

    output = image.copy()

    if top_n is None:
        selected_candidates = candidates
    else:
        selected_candidates = candidates[:top_n]

    for index, candidate in enumerate(selected_candidates):
        x, y, w, h = candidate["box"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            str(index + 1),
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output