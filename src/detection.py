import cv2
import numpy as np

from src.preprocessing import remove_noise, enhance_contrast


def canny_pipeline(gray_image):
    """
    Canny-based pipeline.

    This pipeline is used mainly for comparison.
    It detects general edges, so it may produce many false candidates
    in complex backgrounds.
    """

    denoised = remove_noise(gray_image)
    enhanced = enhance_contrast(denoised)

    median_intensity = np.median(enhanced)

    lower_threshold = int(max(0, 0.70 * median_intensity))
    upper_threshold = int(min(255, 1.30 * median_intensity))

    edges = cv2.Canny(
        enhanced,
        lower_threshold,
        upper_threshold
    )

    # Smaller kernel to avoid connecting floor/background edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (11, 3)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Small dilation only, not aggressive
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    dilated = cv2.dilate(
        closed,
        small_kernel,
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
    
def blackhat_pipeline(gray_image):
    """
    Improved Blackhat + Sobel pipeline for Egyptian license plates.

    Goal:
    - highlight dark text-like regions on bright plate background
    - connect nearby character regions into one plate-like region
    - reduce small noise
    """

    # Step 1: improve image before Blackhat
    denoised = remove_noise(gray_image)
    enhanced = enhance_contrast(denoised)

    # Step 2: blackhat kernel
    # مناسب لشكل اللوحات الأفقية
    rect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (19, 5)
    )

    blackhat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_BLACKHAT,
        rect_kernel
    )

    # Step 3: Sobel X
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

    # Step 4: blur
    blurred = cv2.GaussianBlur(
        grad_x,
        (5, 5),
        0
    )

    # Step 5: close to connect nearby text strokes
    closed = cv2.morphologyEx(
        blurred,
        cv2.MORPH_CLOSE,
        rect_kernel
    )

    # Step 6: threshold
    thresh = cv2.threshold(
        closed,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]

    # Step 7: remove tiny noise
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    opened = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        small_kernel
    )

    # Step 8: connect characters horizontally into plate region
    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (27, 5)
    )

    merged = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        merge_kernel
    )

    # Step 9: mild dilation to strengthen the plate region
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    dilated = cv2.dilate(
        merged,
        dilate_kernel,
        iterations=1
    )

    return {
        "denoised": denoised,
        "enhanced": enhanced,
        "blackhat": blackhat,
        "sobel_x": grad_x,
        "blurred": blurred,
        "closed": closed,
        "threshold": thresh,
        "opened": opened,
        "merged": merged,
        "dilated": dilated
    }

def extract_canny_candidates(binary_image, original_image):
    """
    Extract license plate candidates from Canny output.

    Logic:
    Canny detects many general edges in the image.
    So we use lighter morphology and stricter filtering.
    """

    H, W = original_image.shape[:2]
    image_area = H * W

    binary = binary_image.copy()

    # Step 1: light closing only
    # Canny has many edges, so aggressive closing may connect floor/car parts.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (11, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # Step 2: very small dilation
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    binary = cv2.dilate(
        binary,
        dilate_kernel,
        iterations=1
    )

    # Step 3: find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w == 0 or h == 0:
            continue

        contour_area = cv2.contourArea(contour)
        rect_area = w * h

        if rect_area == 0:
            continue

        aspect_ratio = w / float(h)
        area_ratio = rect_area / float(image_area)
        extent = contour_area / float(rect_area)
        center_y = (y + h / 2) / H

        # Canny needs stricter aspect ratio
        if not (2.0 <= aspect_ratio <= 7.5):
            continue

        # Canny often detects small random areas, so use stronger size limits
        if w < 100 or h < 22:
            continue

        # Candidate should have reasonable size
        if area_ratio < 0.0015:
            continue

        if area_ratio > 0.09:
            continue

        # Canny regions should be more filled than Blackhat regions
        if extent < 0.10:
            continue

        # Avoid very top and very bottom areas
        if not (0.18 <= center_y <= 0.90):
            continue

        # Ignore regions near the floor/bottom
        if y + h > H * 0.94:
            continue

        candidate = {
            "box": (x, y, w, h),
            "aspect_ratio": aspect_ratio,
            "area_ratio": area_ratio,
            "contour_area": contour_area,
            "extent": extent,
            "center_y": center_y,
            "method": "canny"
        }

        candidates.append(candidate)

    return candidates
    
def extract_blackhat_candidates(binary_image, original_image):
    """
    Extract possible Egyptian license plate candidates from Blackhat output.

    This version is designed for:
    - Egyptian plates
    - slightly tilted plates
    - smaller plates
    - lower plate positions
    """

    contours, _ = cv2.findContours(
        binary_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = original_image.shape[:2]
    image_area = H * W

    candidates = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w == 0 or h == 0:
            continue

        rect_area = w * h

        if rect_area == 0:
            continue

        # normal bounding rectangle features
        aspect_ratio = w / float(h)
        area_ratio = rect_area / float(image_area)
        extent = contour_area / float(rect_area)
        center_y = (y + h / 2) / H

        # rotated rectangle for slightly tilted plates
        rotated_rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rotated_rect

        if rw == 0 or rh == 0:
            continue

        rotated_w = max(rw, rh)
        rotated_h = min(rw, rh)
        rotated_aspect_ratio = rotated_w / float(rotated_h)

        # ------------------------
        # Filtering rules
        # ------------------------

        # 1) normal box shape
        if not (1.4 <= aspect_ratio <= 10.0):
            continue

        # 2) rotated box shape (better for tilted plates)
        if not (1.6 <= rotated_aspect_ratio <= 10.5):
            continue

        # 3) allow smaller plates
        if w < 55 or h < 13:
            continue

        # 4) area relative to image
        if not (0.0006 <= area_ratio <= 0.13):
            continue

        # 5) allow lower extent because blackhat regions may be fragmented
        if extent < 0.035:
            continue

        # 6) allow lower plates
        if not (0.12 <= center_y <= 0.97):
            continue

        # 7) don't reject the very bottom too aggressively
        if y + h > H * 0.995:
            continue

        candidate = {
            "box": (x, y, w, h),
            "rotated_rect": rotated_rect,
            "aspect_ratio": aspect_ratio,
            "rotated_aspect_ratio": rotated_aspect_ratio,
            "area_ratio": area_ratio,
            "contour_area": contour_area,
            "extent": extent,
            "center_y": center_y,
            "method": "blackhat"
        }

        candidates.append(candidate)

    return candidates


def draw_candidates(image, candidates, top_n=None):
    """
    Draw candidate boxes.

    Green = Blackhat candidate
    Blue  = Canny candidate
    """

    output = image.copy()

    if top_n is None:
        selected_candidates = candidates
    else:
        selected_candidates = candidates[:top_n]

    for index, candidate in enumerate(selected_candidates):
        x, y, w, h = candidate["box"]

        source = candidate.get("source", candidate.get("method", "unknown"))

        if source == "blackhat":
            color = (0, 255, 0)  # green
        elif source == "canny":
            color = (255, 0, 0)  # blue in BGR
        else:
            color = (0, 255, 255)

        label = f"{index + 1}"

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            output,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output