import cv2
import pytesseract

# Load the preprocessed image (binary image)
image = cv2.imread('processed_image.png')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours in the image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours (from left to right, top to bottom)
sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1]*1000 + cv2.boundingRect(c)[0])

# Create a copy of the image to draw bounding boxes
image_copy = image.copy()

# Loop through each contour to extract numbers
for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Filter small contours that are unlikely to be text
    if w > 10 and h > 10:
        # Crop the contour area from the image
        roi = gray[y:y+h, x:x+w]

        # Optionally, apply additional preprocessing on the cropped region if needed
        # For example, adaptive thresholding:
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Run OCR on the cropped region
        extracted_number = pytesseract.image_to_string(roi, config='--psm 8 -c tessedit_char_whitelist="0123456789"')

        # Draw bounding boxes for visualizing detected text regions
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Print the recognized number for debugging
        if extracted_number:
            print(f"Detected number at ({x}, {y}): {extracted_number.strip()}")

# Save the result to see where OCR is being applied
cv2.imwrite('detected_numbers.png', image_copy)
