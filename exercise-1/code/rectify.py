import cv2
import os
import sys

current_points = []
line_group1 = []
line_group2 = []


def draw_line(event, x, y, flags, param):
    global current_points
    line_points, img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(current_points) == 2:
            pt1, pt2 = current_points
            line_points.append((pt1, pt2))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            print(f"Line added: {pt1} -> {pt2}")
            current_points = []


def main():

    cv2.namedWindow("Select Parallel Lines")
    image_path = "C:\\Users\\lior.kotlar\\Documents\\Lior Studies\\3D-Vision-Course\\exercise-1\\data\\livingroom.jpeg"
    img = cv2.imread(image_path)

    for linegroup in [line_group1, line_group2]:
        imgcpy = img.copy()
        cv2.setMouseCallback("Select Parallel Lines", draw_line, (linegroup, imgcpy))
        while True:
            cv2.imshow("Select Parallel Lines", imgcpy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    print("\nSelected Lines:")

    for i, line in enumerate(line_group1):
        print(f"Line {i + 1}: {line}")

    for i, line in enumerate(line_group2):
        print(f"Line {i + 1}: {line}")

if __name__ == "__main__":
    main()