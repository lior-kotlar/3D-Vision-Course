import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

right_frame_path = 'exercise-2/data/right.jpeg'
left_frame_path = 'exercise-2/data/left.jpeg'

def collect_7_point_correspondences(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images could not be loaded.")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    points_img1 = []
    points_img2 = []

    for i in range(7):
        plt.figure(figsize=(10, 5))
        plt.imshow(img1)
        plt.title(f"Select point {i+1} in Image 1")
        point1 = plt.ginput(1)[0]
        plt.close()
        points_img1.append(point1)

        plt.figure(figsize=(10, 5))
        plt.imshow(img2)
        plt.title(f"Select point {i+1} in Image 2")
        point2 = plt.ginput(1)[0]
        plt.close()
        points_img2.append(point2)

    return points_img1, points_img2

def main():
    points = collect_7_point_correspondences(right_frame_path, left_frame_path)


if __name__ == "__main__":
    main()