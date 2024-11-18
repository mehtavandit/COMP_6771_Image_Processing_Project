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
