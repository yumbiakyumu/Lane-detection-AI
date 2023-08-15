# Import necessary libraries
import matplotlib.pylab as apl
import cv2
import numpy as ny

# Read an image named 'lane9.jpg' using OpenCV
image = cv2.imread('lane9.jpg')
# Convert the image from RGB to BGR color space
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Print the shape (dimensions) of the image
print(image.shape)
# Extract height and width of the image
height = image.shape[0]
width = image.shape[1]

# Define vertices of a polygon that defines a region of interest
important_region_vertices = [
    (0, height),
    (width/1.95, height/1.55),
    (width, height)
]

# Define a function to extract the important region from an image
def important_region(img, vertices):
    # Create a mask of zeros with the same dimensions as the image
    mask = ny.zeros_like(img)
    # Get the number of color channels in the image
    channel_count = img.shape[2]
    # Create a color value to match for filling the mask (white color)
    match_mask_color = (255,) * channel_count
    # Fill the polygon defined by 'vertices' with the match_mask_color
    cv2.fillPoly(mask, vertices, match_mask_color)
    # Perform a bitwise AND operation between the image and the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Get the masked image using the 'important_region' function
masked_image = important_region(image, ny.array([important_region_vertices], ny.int32))

# Display the masked image using Matplotlib
apl.imshow(masked_image)
apl.show()
