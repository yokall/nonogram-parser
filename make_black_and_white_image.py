import cv2
import pytesseract

# Load an example image
image = cv2.imread('nonogram_sample.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

denoised_image = cv2.fastNlMeansDenoising(gray_image, h=30)

# Apply a binary threshold to get a black-and-white effect
_, binary_image = cv2.threshold(denoised_image, 200, 255, cv2.THRESH_BINARY_INV)

# Invert the colors (black becomes white and white becomes black)
binary_image = cv2.bitwise_not(binary_image)

# Save the result for verification (Optional)
cv2.imwrite('processed_image.png', binary_image)
