import numpy as np
import cv2
import argparse
from scipy import ndimage as ndi

def rotate_image(image, rotate_angle):
    if rotate_angle:
        return np.rot90(image, -1) #clockwise (to be checked)
    else:
        return np.rot90(image,1) #anti clockwise (to be checked)


def resize(image, new_height, new_width):
    image = image.astype(np.float64)

    height, width = image.shape[:2]

    if height + new_height <= 0 or width + new_width <=0 or abs(new_height) > height or abs(new_width) > width:
        raise ValueError("Invalid seam dimensions: The image cannot be resized to zero or negative dimensions.")


    result_image = image

    if new_width > 0 :
        result_image = insert_seam(result_image, new_width)
    elif new_width < 0:
        result_image = remove_seam(resized_image, -new_width)


    if new_height > 0:
        result_image = rotate_image(result_image, True)
        result_image = insert_seam(result_image, new_height)
        result_image = rotate_image(result_image, False)

    elif new_height < 0:
        result_image = rotate_image(result_image, True)
        result_image = remove_seam(result_image, -new_height)
        result_image = rotate_image(result_image, False)

    return result_image


if __name__ == '__main__':
    print("Select an option:")
    print("1. Resize image")
    print("2. Object removal")
    option = input("Enter 1 for resize or 2 for object removal: ")

    if option == "1":  # Resize image
        # Ask for image name
        im_path = input("Enter the path to the image: ")

        im = cv2.imread(im_path)
        if im is None:
            print("Error: Image not found!")
            exit()

        new_width = int(input("Enter the new width: "))
        new_height = int(input("Enter the new height: "))

        try:
            resized_image = resize(im, new_height, new_width)
            output_name = input("Enter the output file name: ")
            # cv2.imwrite(output_name, resized_image)
            print(f"Image resized and saved as {output_name}")

        except ValueError as e:
            print("Error Occured")

    # elif option == "2":  # Object removal
    #     # Ask for image and mask paths
    #     im_path = input("Enter the path to the image: ")
    #     mask_path = input("Enter the path to the protective mask (optional, press Enter to skip): ")
    #     rmask_path = input("Enter the path to the removal mask: ")
    #
    #     # Load the image and masks
    #     im = cv2.imread(im_path)
    #     if im is None:
    #         print("Error: Image not found!")
    #         exit()
    #
    #     mask = cv2.imread(mask_path, 0) if mask_path else None
    #     rmask = cv2.imread(rmask_path, 0)
    #
    #     # Ask for visualization and horizontal seam removal options
    #     visualize = input("Do you want to visualize the seam removal process? (y/n): ").lower() == 'y'
    #     hremove = input("Remove horizontal seams for object removal? (y/n): ").lower() == 'y'
    #
    #     # Perform object removal
    #     output_name = input("Enter the output file name: ")
    #     # output = object_removal(im, rmask, mask, visualize, hremove)
    #     # cv2.imwrite(output_name, output)
    #     print(f"Object removal complete and saved as {output_name}")

    else:
        print("Invalid option! Please enter 1 or 2.")
