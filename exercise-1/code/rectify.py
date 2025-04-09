import cv2
import os
import sys

current_points = []
lines = []
image_path = "C:\\Users\\lior.kotlar\\Documents\\Lior Studies\\3D-Vision-Course\\exercise-1\\data\\livingroom.jpeg"
img = cv2.imread(image_path)

def draw_line(event, x, y, flags, param):
    global current_points, lines, img

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(current_points) == 2:
            pt1, pt2 = current_points
            lines.append((pt1, pt2))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            print(f"Line added: {pt1} -> {pt2}")
            current_points = []


def select_points(image):
    while True:
        cv2.imshow("Select Parallel Lines", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

def main():

    cv2.namedWindow("Select Parallel Lines")

    cv2.setMouseCallback("Select Parallel Lines", draw_line)
    while True:
        cv2.imshow("Select Parallel Lines", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    print("\nSelected Lines:")

    for i, line in enumerate(lines):
        print(f"Line {i + 1}: {line}")

if __name__ == "__main__":
    main()