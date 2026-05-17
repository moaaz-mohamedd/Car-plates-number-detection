import cv2
import numpy as np

from src.preprocessing import remove_noise, enhance_contrast


  
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