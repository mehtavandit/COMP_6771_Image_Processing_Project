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

## Functions Overview

### Main Functions

- **`resize_and_save(image, new_width, new_height, im_path)`**:  
  Resizes the image to the target dimensions and saves the result.

- **`resize(image, new_height, new_width)`**:  
  Resizes the image using seam carving for width and height adjustments.



### Seam Carving Functions

- **`energy(image)`**:  
  Computes the energy map of the image based on gradients.

- **`get_minimum_seam(image)`**:  
  Finds the vertical seam with the lowest energy using dynamic programming.

- **`remove_seam(image, seam_idx)`**:  
  Removes a vertical seam from the image.

- **`insert_seam(image, seam_idx)`**:  
  Inserts a vertical seam into the image.

- **`seams_removal(image, num_remove)`**:  
  Removes multiple seams to reduce the image width.

- **`seams_insertion(image, num_add)`**:  
  Inserts multiple seams to increase the image width.



### Utility Functions

- **`rotate_image(image, rotate_angle)`**:  
  Rotates the image 90 degrees clockwise or anti-clockwise.

## Example Usage

### Resize Image to Specific Dimensions

1. Run the program:
   ```bash
   python seam_carving.py
   ```
2. Enter the image name:
   ```bash
   Enter the file name: example.jpg
   ```
3. Choose option 1 for single resizing:
   ```bash
   Enter 1 for resize or 2 for multi-size images: 1
   ```
4. Enter target dimensions:
   ```bash
   Enter the new width: 300
   Enter the new height: 200
   ```
5. The resized image will be saved as example_300x200.jpg.

### Resize Multiple Images

1. Choose option 2 for batch resizing:
   ```bash
   Enter 1 for resize or 2 for multi-size images: 2
   ```
2. Provide dimensions iteratively:
   ```bash
   Enter the new width (or -1 to exit): 400
   Enter the new height (or -1 to exit): 300
   ```
3. Exit by entering -1 for width:
   ```bash
   Enter the new width (or -1 to exit): -1
   ```

## Results

### Resize Image to Specific Dimensions

1. Original test1.jpg Image (800x600)
   
   ![Original Image](Test1/test1.jpg)
   
2. Scaled up test1.jpg with dimension (900x700)

   ![Resized Image](Test1/test1_900x700.jpg)

3. Scaled down test1.jpg with dimensions (700x500)

    ![Resized Image](Test%202/test2_700x500.jpg)


   
### Resize Multiple Images


1. Original castle.jpg Image (1428x968)

   ![Original Image](Multi-Size/castle.jpg)

2. Resized castle.jpg (800x800)

   ![Resized Image](Multi-Size/castle_800x800.jpg)

3. Resized castle.jpg (1600x800)

   ![Resized Image](Multi-Size/castle_1600x800.jpg)

4. Resized castle.jpg (1600x1600)

   ![Resized Image](Multi-Size/castle_1600x1600.jpg)

5. Resized castle.jpg (1000x1000)

   ![Resized Image](Multi-Size/castle_1000x1000.jpg)


## Contributors

| **Team Members**        | **Student ID** | **Email**                  |
|-------------------------|----------------|----------------------------|
| Vandit Mehta            | 40232414       | mehtavandit2205@gmail.com |
| Devshree Shah             | 40269569       | shahdevshree07@gmail.com  |


| **Lecturer**            | **Email**                  |
|-------------------------|----------------------------|
| Dr Yiming Xiao  | yiming.xiao@concordia.ca |
