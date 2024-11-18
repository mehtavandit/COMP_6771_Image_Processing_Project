# Seam Carving Image Resizer

This Python project implements an image resizing tool using the **Seam Carving** technique. Seam carving is a content-aware image resizing algorithm that adjusts image dimensions while preserving important features and minimizing distortion.

## Features
- Resize images to specified dimensions while preserving their content.
- Supports both single resize and batch resizing modes.
- Handles both width and height resizing using seam carving.
- Outputs resized images with a unique filename based on their new dimensions.

## Prerequisites

Make sure you have the following installed on your system:

1. **Python 3.x**
2. Required Python libraries:
   - `numpy`
   - `opencv-python`
   - `scipy`

You can install the required libraries using:
```bash
pip install numpy opencv-python scipy
```

## How to Use

### Clone the Repository
```bash
git clone https://github.com/mehtavandit/COMP_6771_Image_Processing_Project
```

## Run the program
```bash
python seam_carving.py
```

### Input Image and Options

1. Enter the image file name (ensure the image is in the same directory or provide the full path).
2. Select one of the options:
   - **Option 1**: Resize the image to a single specified width and height.
   - **Option 2**: Perform multiple resizing operations with different dimensions.

### Outputs

- Resized images are saved in the current working directory with a filename format: `original_name_<width>x<height>.jpg`.
- Resized images are also displayed for preview.
