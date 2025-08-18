import cv2
import numpy as np
import os

video_path = 'exercise-2/data/pen_drop.mp4'
save_directory = 'exercise-2/data/'
def get_paths():
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame1_path = os.path.join(save_directory, f"{video_name}_frame1.png")
    frame2_path = os.path.join(save_directory, f"{video_name}_frame2.png")
    return frame1_path, frame2_path

def get_time_between_frames():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_between_frames = 1.0 / fps
    print(f"Time between frames: {time_between_frames:.4f} seconds")
    return time_between_frames

def extract_two_frames(frame1_path, frame2_path):
    os.makedirs(save_directory, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return
    ret1, frame1 = cap.read()
    index = 0
    while ret1:
        ret2, frame2 = cap.read()
        if not ret2:
            break
        combined = cv2.hconcat([frame1, frame2])
        cv2.imshow('Frame pair (press s to save, q to quit)', combined)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cv2.imwrite(frame1_path, frame1)
            cv2.imwrite(frame2_path, frame2)
            print(f"Saved frames to {frame1_path} and {frame2_path}")
            break
        elif key == ord('q'):
            print("User quit without saving.")
            break

        frame1 = frame2
        ret1 = ret2
        index += 1
    cap.release()
    cv2.destroyAllWindows()


def select_2points_per_frame(frame1, frame2):
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Points", param)

    frame1_copy = frame1.copy()
    cv2.imshow("Select Points", frame1_copy)
    cv2.setMouseCallback("Select Points", click_event, frame1_copy)
    print("Select 2 points in Frame 1...")
    while len(points) < 2:
        if cv2.waitKey(1) == 27:  # ESC key to cancel
            print("Cancelled.")
            cv2.destroyAllWindows()
            return None
    cv2.waitKey(500)

    # --- Frame 2 ---
    frame2_copy = frame2.copy()
    cv2.imshow("Select Points", frame2_copy)
    cv2.setMouseCallback("Select Points", click_event, frame2_copy)
    print("Select 2 points in Frame 2...")
    while len(points) < 4:
        if cv2.waitKey(1) == 27:
            print("Cancelled.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return points[:2], points[2:]


def calculate_ttc_from_points(points, time_between_frames_secs):
    frame1_points, frame2_points = points
    object_length1 = np.linalg.norm(np.array(frame1_points[1]) - np.array(frame1_points[0]))
    object_length2 = np.linalg.norm(np.array(frame2_points[1]) - np.array(frame2_points[0]))
    print(f'object_length1: {object_length1}')
    print(f'object_length2: {object_length2}')
    length_change = object_length2 - object_length1
    print(f'length_change: {length_change}')
    change_velocity = length_change / time_between_frames_secs
    print(f'change_velocity: {change_velocity}')
    ttc = object_length2 / change_velocity
    print(f'ttc: {ttc}')
    return ttc


def get_frames_from_paths(frame1_path, frame2_path):
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)
    return frame1, frame2

def main():
    frame1_path, frame2_path = get_paths()
    if not os.path.exists(frame1_path) or not os.path.exists(frame2_path):
        extract_two_frames(frame1_path, frame2_path)
    time_between_frames = get_time_between_frames()
    frame1, frame2 = get_frames_from_paths(frame1_path, frame2_path)
    points = select_2points_per_frame(frame1, frame2)
    ttc = calculate_ttc_from_points(points, time_between_frames)



if __name__ == "__main__":
    main()
