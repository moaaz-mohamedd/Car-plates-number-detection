import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image
from src.detection import (
    blackhat_pipeline,
    extract_blackhat_candidates,
    draw_candidates
)
from src.scoring import (
    rank_blackhat_candidates,
    get_top_candidates
)
from src.utils import bgr_to_rgb, crop_with_padding


st.set_page_config(
    page_title="Car Plate Detection",
    page_icon="🚗",
    layout="wide"
)


st.title("🚗 Car Plate Detection Using Image Processing")
st.write(
    "This app detects possible car plate regions using classical image processing techniques: "
    "preprocessing, Blackhat morphology, contour extraction, and simple candidate scoring."
)


uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

show_steps = st.checkbox(
    "Show image processing steps",
    value=True
)

show_all_candidates = st.checkbox(
    "Show all extracted candidates",
    value=True
)

top_n = st.slider(
    "Number of top candidates to show",
    min_value=1,
    max_value=5,
    value=3
)


if uploaded_file is not None:
    # -----------------------------
    # 1. Read uploaded image
    # -----------------------------
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)

    # OpenCV works with BGR
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    # -----------------------------
    # 2. Preprocessing
    # -----------------------------
    preprocessed = preprocess_image(
        image_bgr,
        width=900
    )

    resized = preprocessed["resized"]
    gray = preprocessed["gray"]

    # -----------------------------
    # 3. Blackhat pipeline
    # -----------------------------
    blackhat_output = blackhat_pipeline(gray)

    # -----------------------------
    # 4. Extract candidates
    # -----------------------------
    blackhat_candidates = extract_blackhat_candidates(
        blackhat_output["dilated"],
        resized
    )

    # -----------------------------
    # 5. Score candidates
    # -----------------------------
    ranked_candidates = rank_blackhat_candidates(
        blackhat_candidates,
        blackhat_output["dilated"],
        resized.shape
    )

    top_candidates = get_top_candidates(
        ranked_candidates,
        top_n=top_n
    )

    # -----------------------------
    # 6. Display main result
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image_rgb,
            use_container_width=True
        )

    with col2:
        st.subheader(f"Top {top_n} Candidates")
        top_candidates_image = draw_candidates(
            resized,
            top_candidates
        )

        st.image(
            bgr_to_rgb(top_candidates_image),
            use_container_width=True
        )

    # -----------------------------
    # 7. Best candidate crop
    # -----------------------------
    st.divider()

    if len(top_candidates) > 0:
        best_candidate = top_candidates[0]

        best_crop = crop_with_padding(
            resized,
            best_candidate["box"],
            padding=10
        )

        st.subheader("Best Detected Plate Candidate")

        col3, col4 = st.columns([1, 1])

        with col3:
            st.image(
                bgr_to_rgb(best_crop),
                caption="Best Candidate Crop",
                use_container_width=True
            )

        with col4:
            st.write("### Best Candidate Info")
            st.write(f"**Score:** {best_candidate['score']:.3f}")
            st.write(f"**Box:** {best_candidate['box']}")
            st.write(f"**Aspect Ratio:** {best_candidate['aspect_ratio']:.2f}")
            st.write(f"**Area Ratio:** {best_candidate['area_ratio']:.4f}")
            st.write(f"**Width Ratio:** {best_candidate['width_ratio']:.3f}")
            st.write(f"**Height Ratio:** {best_candidate['height_ratio']:.3f}")
            st.write(f"**White Density:** {best_candidate['white_density']:.3f}")
            st.write(f"**Center Y:** {best_candidate['center_y']:.2f}")

    else:
        st.error("No plate candidates detected. Try another image.")

    # -----------------------------
    # 8. Show top candidate crops
    # -----------------------------
    if len(top_candidates) > 0:
        st.divider()
        st.subheader(f"Top {top_n} Candidate Crops")

        cols = st.columns(len(top_candidates))

        for i, candidate in enumerate(top_candidates):
            crop = crop_with_padding(
                resized,
                candidate["box"],
                padding=10
            )

            with cols[i]:
                st.image(
                    bgr_to_rgb(crop),
                    caption=f"Candidate {i + 1} | Score: {candidate['score']:.3f}",
                    use_container_width=True
                )

    # -----------------------------
    # 9. Show all candidates
    # -----------------------------
    if show_all_candidates:
        st.divider()
        st.subheader("All Extracted Blackhat Candidates")

        all_candidates_image = draw_candidates(
            resized,
            blackhat_candidates
        )

        st.image(
            bgr_to_rgb(all_candidates_image),
            use_container_width=True
        )

        st.write(f"**Number of extracted candidates:** {len(blackhat_candidates)}")

    # -----------------------------
    # 10. Show processing steps
    # -----------------------------
    if show_steps:
        st.divider()
        st.subheader("Image Processing Steps")

        step_col1, step_col2 = st.columns(2)

        with step_col1:
            st.image(
                preprocessed["gray"],
                caption="Grayscale",
                use_container_width=True,
                clamp=True
            )

            st.image(
                blackhat_output["blackhat"],
                caption="Blackhat",
                use_container_width=True,
                clamp=True
            )

            st.image(
                blackhat_output["threshold"],
                caption="Threshold",
                use_container_width=True,
                clamp=True
            )

        with step_col2:
            st.image(
                blackhat_output["sobel_x"],
                caption="Sobel X",
                use_container_width=True,
                clamp=True
            )

            st.image(
                blackhat_output["merged"],
                caption="Merged Regions",
                use_container_width=True,
                clamp=True
            )

            st.image(
                blackhat_output["dilated"],
                caption="Final Dilated Binary",
                use_container_width=True,
                clamp=True
            )

else:
    st.info("Upload a car image to start detection.")