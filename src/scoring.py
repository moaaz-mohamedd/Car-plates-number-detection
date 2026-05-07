import numpy as np


def closeness_score(value, ideal, tolerance):
    """
    Calculate how close a value is to an ideal value.

    Score = 1 means very close to ideal.
    Score = 0 means far from ideal.

    Example:
    If ideal aspect ratio is 4.0,
    a candidate with aspect ratio 4.1 gets a high score.
    """

    score = 1 - abs(value - ideal) / tolerance
    return max(0, min(1, score))


def score_blackhat_candidate(candidate, binary_image, image_shape):
    """
    Score one Blackhat candidate.

    The scoring is based on simple handcrafted features:
    1. Width ratio
    2. Height ratio
    3. Area ratio
    4. Aspect ratio
    5. Rotated aspect ratio
    6. White density
    7. Extent
    8. Position

    This is not deep learning.
    It is a simple rule-based scoring system.
    """

    x, y, w, h = candidate["box"]
    H, W = image_shape[:2]

    aspect_ratio = candidate["aspect_ratio"]
    rotated_aspect_ratio = candidate.get(
        "rotated_aspect_ratio",
        aspect_ratio
    )

    area_ratio = candidate["area_ratio"]
    extent = candidate["extent"]
    center_y = candidate["center_y"]

    width_ratio = w / float(W)
    height_ratio = h / float(H)

    roi = binary_image[y:y + h, x:x + w]

    if roi.size == 0:
        white_density = 0
    else:
        white_density = np.mean(roi > 0)

    # --------------------------------------------------
    # 1. Shape scores
    # --------------------------------------------------
    # License plates are usually horizontal rectangles.
    # Egyptian plates may vary, so we keep tolerance flexible.

    aspect_score = closeness_score(
        aspect_ratio,
        ideal=3.8,
        tolerance=2.2
    )

    rotated_aspect_score = closeness_score(
        rotated_aspect_ratio,
        ideal=4.0,
        tolerance=2.5
    )

    # --------------------------------------------------
    # 2. Size scores
    # --------------------------------------------------
    # These are important because wrong objects may be rectangular
    # but their size is usually not close to a real plate.

    width_score = closeness_score(
        width_ratio,
        ideal=0.18,
        tolerance=0.14
    )

    height_score = closeness_score(
        height_ratio,
        ideal=0.06,
        tolerance=0.04
    )

    area_score = closeness_score(
        area_ratio,
        ideal=0.012,
        tolerance=0.015
    )

    # --------------------------------------------------
    # 3. Density / fill scores
    # --------------------------------------------------
    # Blackhat output should contain enough white pixels inside the plate area,
    # but not be completely white.

    density_score = closeness_score(
        white_density,
        ideal=0.28,
        tolerance=0.22
    )

    extent_score = closeness_score(
        extent,
        ideal=0.45,
        tolerance=0.35
    )

    # --------------------------------------------------
    # 4. Position score
    # --------------------------------------------------
    # Keep this flexible because plates can appear low or slightly off-center.

    position_score = closeness_score(
        center_y,
        ideal=0.65,
        tolerance=0.35
    )

    # --------------------------------------------------
    # Final score
    # --------------------------------------------------
    # Size has the biggest weight because it helps reject random objects.
    # Shape is also important.
    # Position has low weight so we do not reject lower plates.

    final_score = (
        width_score * 0.24 +
        height_score * 0.18 +
        area_score * 0.18 +
        aspect_score * 0.16 +
        rotated_aspect_score * 0.10 +
        density_score * 0.08 +
        extent_score * 0.04 +
        position_score * 0.02
    )

    scored_candidate = candidate.copy()

    scored_candidate["score"] = final_score

    scored_candidate["width_ratio"] = width_ratio
    scored_candidate["height_ratio"] = height_ratio
    scored_candidate["white_density"] = white_density

    scored_candidate["width_score"] = width_score
    scored_candidate["height_score"] = height_score
    scored_candidate["area_score"] = area_score
    scored_candidate["aspect_score"] = aspect_score
    scored_candidate["rotated_aspect_score"] = rotated_aspect_score
    scored_candidate["density_score"] = density_score
    scored_candidate["extent_score"] = extent_score
    scored_candidate["position_score"] = position_score

    return scored_candidate


def rank_blackhat_candidates(candidates, binary_image, image_shape):
    """
    Rank Blackhat candidates from best to worst.

    Input:
        candidates:
            List of candidates extracted from extract_blackhat_candidates()

        binary_image:
            Usually blackhat_output["dilated"]

        image_shape:
            resized.shape

    Output:
        List of candidates sorted by score descending.
    """

    scored_candidates = []

    for candidate in candidates:
        scored_candidate = score_blackhat_candidate(
            candidate,
            binary_image,
            image_shape
        )

        scored_candidates.append(scored_candidate)

    ranked_candidates = sorted(
        scored_candidates,
        key=lambda c: c["score"],
        reverse=True
    )

    return ranked_candidates


def get_top_candidates(ranked_candidates, top_n=3):
    """
    Return the top N ranked candidates.
    """

    return ranked_candidates[:top_n]