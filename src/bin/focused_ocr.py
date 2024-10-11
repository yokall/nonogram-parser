from collections import namedtuple
import cv2
import pytesseract
from pathlib import Path

script_dir = Path(__file__).parent.resolve()

# Load the preprocessed image (binary image)
image = cv2.imread(f"{script_dir}/../../tmp/processed_image.png")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours in the image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours (from left to right, top to bottom)
sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1]*1000 + cv2.boundingRect(c)[0])

# Create a copy of the image to draw bounding boxes
image_copy = image.copy()

Point = namedtuple('Point', ['x', 'y', 'n'])

# Initialize an empty list to hold the points
coordinates = []
number_width = None

# Loop through each contour to extract numbers
for contour in sorted_contours:
    x, y, w, h = cv2.boundingRect(contour)

    # There will never be a clue at position 0,0
    if x == 0 and y ==0:
        continue

    # Filter small contours that are unlikely to be text
    if w > 10 and h > 10:
        # Crop the contour area from the image
        roi = image[y:y+h, x:x+w]

        # Run OCR on the cropped region
        extracted_number = pytesseract.image_to_string(roi, config='--psm 8 -c tessedit_char_whitelist="0123456789"')

        # Print the recognized number for debugging
        if extracted_number:
            # Draw bounding boxes for visualizing detected text regions
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 5)

            point = Point(x, y, extracted_number)

            # Append the new point to the list
            coordinates.append(point)

            if number_width is None:
                number_width = w

# Save the result to see where OCR is being applied
cv2.imwrite(f"{script_dir}/../../tmp/detected_numbers.png", image_copy)

x_split = coordinates[-1].x + number_width

sorted_coordinates = sorted(coordinates, key=lambda point: (point.y, point.x), reverse=True)

for point in sorted_coordinates:
    print(point)

print("\nSplit x pos is: " + str(x_split))

# Split the sorted list into two lists based on the x value
column_clues = [point for point in sorted_coordinates if point.x >= x_split]
row_clues = [point for point in sorted_coordinates if point.x < x_split]

column_clues = sorted(column_clues, key=lambda point: (point.x, point.y))
row_clues = sorted(row_clues, key=lambda point: (point.y, point.x))

print("\nColumn clues:")
for point in column_clues:
    print(point)

print("\nRow clues:")
for point in row_clues:
    print(point)
