# Car Plate Detection Using Classical Image Processing

A classical image processing project for detecting car license plate candidates from vehicle images, with a focus on Egyptian car plates.  
The system uses preprocessing, Blackhat morphology, contour extraction, and simple rule-based scoring to detect and rank possible plate regions.

---

## Project Overview

This project aims to detect car license plates from input images using traditional image processing techniques without using deep learning models.

The main goal is to build a complete image processing pipeline that can:

- Read and preprocess car images.
- Enhance image quality.
- Highlight plate-like regions.
- Extract possible license plate candidates.
- Rank candidates using simple handcrafted features.
- Display the top detected plate regions through a Streamlit web interface.

---

## Main Idea

Car plates usually contain dark text or numbers on a brighter background.  
To detect these regions, the system mainly uses the Blackhat morphological operation, which helps highlight dark objects on light backgrounds.

The pipeline focuses on extracting plate-like regions based on:

- Shape
- Size
- Aspect ratio
- Area ratio
- White pixel density
- Candidate position
- Rotated rectangle features for slightly tilted plates

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- Pillow

---

## Project Structure

```text
Car-plates-number-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── detection.py
│   ├── scoring.py
│   └── utils.py
│
├── notebooks/
│   └── testing_pipeline.ipynb
│
├── test_images/
│   ├── car1.jpg
│   ├── car2.jpg
│   └── car3.jpg
│
└── assets/
    └── screenshots/
        ├── original_image.png
        ├── preprocessing_steps.png
        ├── blackhat_steps.png
        ├── all_candidates.png
        ├── top_3_candidates.png
        └── best_plate_crop.png

## Image Processing Pipeline

The system follows a complete classical image processing pipeline to detect possible car plate regions from an input vehicle image.

```text
Input Image
↓
Resize Image
↓
Convert to Grayscale
↓
Noise Removal
↓
Contrast Enhancement
↓
Blackhat Morphology
↓
Sobel X Gradient
↓
Thresholding
↓
Morphological Merging
↓
Contour Detection
↓
Candidate Extraction
↓
Candidate Scoring
↓
Top 3 Plate Candidates
```

---

## 1. Preprocessing

The preprocessing stage prepares the image before detecting the plate.

### Steps

- Resize image to a fixed width.
- Convert image to grayscale.
- Apply noise removal.
- Enhance contrast using CLAHE.

### Why Preprocessing Is Important

Different images may have different sizes, lighting conditions, and noise levels.

Preprocessing helps make the detection pipeline more stable and improves the visibility of plate details before applying the detection algorithm.

---

## 2. Blackhat-Based Detection

The main detection method is based on Blackhat morphology.

### Why Blackhat?

Blackhat is useful for detecting dark text-like regions on bright backgrounds.

Since license plates usually contain dark numbers or letters on a lighter plate area, Blackhat helps highlight these regions and makes them easier to extract.

### Blackhat Pipeline

```text
Grayscale Image
↓
Noise Removal
↓
Contrast Enhancement
↓
Blackhat Operation
↓
Sobel X Gradient
↓
Gaussian Blur
↓
Morphological Closing
↓
Thresholding
↓
Region Merging
↓
Dilation
```

---

## 3. Candidate Extraction

After generating the binary image, contours are extracted using OpenCV.

Each contour is converted into a bounding box and filtered using simple rules that describe the expected shape and size of a license plate.

### Candidate Filtering Features

- Candidate width
- Candidate height
- Aspect ratio
- Area ratio
- Extent
- Vertical position
- Rotated rectangle aspect ratio

These rules help remove irrelevant objects and keep only regions that may look like a license plate.

---

## 4. Candidate Scoring

After extracting possible candidates, each candidate is ranked using a simple rule-based scoring function.

The purpose of the scoring stage is to rank the detected regions and select the most plate-like candidates.

### Scoring Features

- Width ratio
- Height ratio
- Area ratio
- Aspect ratio
- Rotated aspect ratio
- White pixel density
- Extent
- Position

After scoring, the system displays the top 3 candidate regions that are most likely to contain the car plate.

---

## 5. Streamlit Web App

The project includes a simple Streamlit interface that allows the user to upload a car image and view the detection results visually.

### App Features

- Upload a car image.
- Display the original image.
- Show Blackhat processing steps.
- Display all extracted candidates.
- Display the top ranked candidates.
- Show the best detected plate crop.
- Display candidate information such as score, aspect ratio, area ratio, and white density.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Car-plates-number-detection.git
cd Car-plates-number-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Requirements

```bash
python -m pip install -r requirements.txt
```

---

## Requirements

```txt
opencv-python
numpy
matplotlib
streamlit
pillow
ipykernel
notebook
```

---

## How to Run the Streamlit App

Run the following command from the project root directory:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal.

Example:

```text
http://localhost:8501
```