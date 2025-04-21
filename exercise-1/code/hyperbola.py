import cv2
import numpy as np


hyperbola_points = []
points_counter = 0


def on_click(event, x, y, flags, param):
    global hyperbola_points, points_counter
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        hyperbola_points.append((x, y))
        points_counter += 1
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


def collect_hyperbola_points(img):
    cv2.namedWindow("select hyperbola points")
    imgcpy = img.copy()
    imgcpy = cv2.resize(imgcpy, None, fx=0.5, fy=0.5)
    cv2.setMouseCallback("select hyperbola points", on_click, param=imgcpy)
    while 1:
        cv2.imshow("select hyperbola points", imgcpy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if points_counter > 5:
                break
            else:
                print("CHOOSE AT LEAST 6 POINTS")



def create_matrix_from_points():
    m = []
    for point in hyperbola_points:
        x, y = point
        m.append([x**2, x*y, y**2, x, y, 1])

    m = np.array(m)

    return m


def get_conic_section_coffs(m):
    _, _, Vt = np.linalg.svd(m)
    coeffs = Vt[-1, :]
    return coeffs


def check_discriminant(coeffs):
    a, b, c, d, e, f = coeffs
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        return True


def main():
    image_path = ".\\exercise-1\\data\\trash2.jpeg"
    img = cv2.imread(image_path)
    collect_hyperbola_points(img)
    m = create_matrix_from_points()
    conic_section = get_conic_section_coffs(m)
    is_hyperbola = check_discriminant(conic_section)
    print(f'hyperbola: {is_hyperbola}')



if __name__ == '__main__':
    main()