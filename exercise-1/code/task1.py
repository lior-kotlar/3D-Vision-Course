import cv2
import numpy as np

BASE_REFERENCE_OBJECT = 0
TOP_REFERENCE_OBJECT = 1
TOP_TARGET_OBJECT_PROJECTION = 2

object_points = []
point_counter = 0


def on_click(event, x, y, flags, param):
    global object_points, point_counter
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        object_points.append((x, y))
        point_counter += 1
        print(point_counter)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


def collect_object_points(img):
    cv2.namedWindow("select points")
    title_refernce = 'reference'
    title_target = 'target'
    imgcpy = img.copy()
    imgcpy = cv2.resize(imgcpy, None, fx=0.25, fy=0.25)
    cv2.setMouseCallback("select points", on_click, param=imgcpy)
    while 1:
        if point_counter < 3:
            window_name = f'select {title_refernce} object points'
        else:
            window_name = f'select {title_target} object points'
        cv2.imshow("select points", imgcpy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if point_counter == 3:
                break
            else:
                print("CHOOSE 3 POINTS")

    return


def get_float_input():
    while True:
        try:
            user_input = input('Enter the reference object\'s height')
            value = float(user_input)
            return value
        except ValueError:
            print('Please enter a valid number.')


def calculate_cross_ratio():
    ref_object_real_height = get_float_input()
    ref_object_pixel_height = np.linalg.norm(np.array(object_points[BASE_REFERENCE_OBJECT]) - np.array(object_points[TOP_REFERENCE_OBJECT]))
    target_object_pixel_height = np.linalg.norm(np.array(object_points[BASE_REFERENCE_OBJECT]) - np.array(object_points[TOP_TARGET_OBJECT_PROJECTION]))
    print(ref_object_pixel_height, target_object_pixel_height, ref_object_real_height)
    real_target_height = (target_object_pixel_height/ref_object_pixel_height) * ref_object_real_height
    return real_target_height


def main():
    # get_float_input()
    image_path = ".\\exercise-1\\data\\drawing of task 1.jpg"
    img = cv2.imread(image_path)
    collect_object_points(img)
    print(object_points)
    target_object_height = calculate_cross_ratio()
    print(target_object_height)


if __name__ == '__main__':
    main()