import cv2
import numpy as np
from scipy import ndimage as ndi


# Function to compute the energy of an image
def compute_energy(image):

    kernel = np.array([1,0,-1])

    x_gradient = ndi.convolve1d(image, kernel, axis=1, mode='wrap')
    y_gradient = ndi.convolve1d(image, kernel, axis=0, mode='wrap')

    gradient_magnitude = np.sqrt(np.square(x_gradient).sum(axis=2) + np.square(y_gradient).sum(axis=2))

    return gradient_magnitude


# Load your seam-carved image (replace 'your_seam_carved_image.jpg' with your actual image path)
your_seam_carved_image = cv2.imread('test1_1000x1000.jpg')

# Load the benchmark seam-carved image (replace 'benchmark_seam_carved_image.jpg' with the actual benchmark path)
benchmark_seam_carved_image = cv2.imread('test1_1000x1000_benchmark.jpg')

# Compute the energy maps for both images
energy_your_algorithm = compute_energy(your_seam_carved_image)
energy_benchmark = compute_energy(benchmark_seam_carved_image)

# Normalize the energy maps to the range 0-255 for visualization
energy_your_algorithm = cv2.normalize(energy_your_algorithm, None, 0, 255, cv2.NORM_MINMAX)
energy_benchmark = cv2.normalize(energy_benchmark, None, 0, 255, cv2.NORM_MINMAX)

# Convert the energy maps to uint8 for display
energy_your_algorithm = np.uint8(energy_your_algorithm)
energy_benchmark = np.uint8(energy_benchmark)

# Show the energy maps
cv2.imshow("Energy (Your Algorithm)", energy_your_algorithm)
cv2.imshow("Energy (Benchmark Algorithm)", energy_benchmark)

# Wait for any key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
