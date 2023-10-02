# Import necessary libraries
import matplotlib.pylab as apl
import cv2
import numpy as ny



# Read an image named 'lane9.jpg' using OpenCV
image = cv2.imread('lane9.jpg')
# Convert the image from RGB to BGR color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    #channel_count = img.shape[2]
    # Create a color value to match for filling the mask (white color)
    match_mask_color = 255
    # Fill the polygon defined by 'vertices' with the match_mask_color
    cv2.fillPoly(mask, vertices, match_mask_color)
    # Perform a bitwise AND operation between the image and the mask
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def draw_lines(image, lines):
    image = ny.copy(image)
    blank_image = ny.zeros((image.shape[0],image.shape[1], 3), dtype=ny.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1,), (x2,y2), (0,255, 0), thickness=3)

    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    return
grayscale_image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image=cv2.Canny(grayscale_image, 100, 200)
# Get the masked image using the 'important_region' function
masked_image = important_region(canny_image, ny.array([important_region_vertices], ny.int32))
lines = cv2.HoughLinesP(masked_image,rho=6,theta=ny.pi/60,threshold=160,lines=ny.array([]),minLineLength=40,maxLineGap=25)

# Display the masked image using Matplotlib
image_with_lines = draw_lines(image, lines)
apl.imshow(image_with_lines)
apl.show()

