import cv2
import pytesseract

# Load an example image
image = cv2.imread('nonogram_sample.jpg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to get a black-and-white effect
_, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)

# Save the result for verification (Optional)
cv2.imwrite('processed_image.png', binary_image)

# Run OCR on the processed image
clues_text = pytesseract.image_to_string(binary_image, config='--psm 6')

# Print the recognized clues
print(clues_text)
