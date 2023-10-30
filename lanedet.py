import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]),
                            minLineLength=40, maxLineGap=100)

    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                if slope < 0 and x1 < width / 2 and x2 < width / 2:  # Left lane
                    left_lane_lines.append((x1, y1))
                    left_lane_lines.append((x2, y2))
                elif slope > 0 and x1 > width / 2 and x2 > width / 2:  # Right lane
                    right_lane_lines.append((x1, y1))
                    right_lane_lines.append((x2, y2))

    if left_lane_lines and right_lane_lines:
        left_lane = np.polyfit([point[0] for point in left_lane_lines], [point[1] for point in left_lane_lines], 1)
        right_lane = np.polyfit([point[0] for point in right_lane_lines], [point[1] for point in right_lane_lines], 1)

        lane_center_x = (left_lane[1] + right_lane[1]) / 2
        lane_width = abs(right_lane[1] - left_lane[1])
        frame_center_x = width / 2

        deviation = frame_center_x - lane_center_x

        if abs(deviation) > lane_width / 4:  # Lane departure alert triggered if more than quarter width is crossed
            cv2.putText(image, "Lane Departure Alert!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        image_with_lines = draw_the_lines(image, [np.array([[int((height - left_lane[1]) / left_lane[0]), height,
                                                             int((height / 2 + 40 - left_lane[1]) / left_lane[0]),
                                                             int(height / 2 + 40)]], dtype=np.int32),
                                                 np.array([[int((height - right_lane[1]) / right_lane[0]), height,
                                                             int((height / 2 + 40 - right_lane[1]) / right_lane[0]),
                                                             int(height / 2 + 40)]], dtype=np.int32)])
        
        return image_with_lines

    else:
        return image  # If no lanes are detected, return the original image

cap = cv2.VideoCapture('test1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
