import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Stereo camera outputs a side-by-side 2560x720 frame.
# We use the left half for pose tracking (right half available for depth later).
STEREO_FRAME_WIDTH = 2560
SINGLE_WIDTH = STEREO_FRAME_WIDTH // 2  # 1280


def split_stereo(frame):
    """Split a 2560x720 side-by-side stereo frame into left and right 1280x720 frames."""
    return frame[:, :SINGLE_WIDTH], frame[:, SINGLE_WIDTH:]


def calculate_angle(a, b, c):
    """Angle at joint b given three keypoints (each a [x, y] array)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class PoseDetector:
    def __init__(self, mock=False, stereo=True):
        self.mock = mock
        self.stereo = stereo
        if not mock:
            self.pose = mp_pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def process(self, frame):
        """Return (annotated_left_frame, landmarks_dict) or (frame, None) on failure."""
        if self.mock:
            return frame, None

        # split stereo frame; use left lens for pose, keep right for future depth use
        if self.stereo and frame.shape[1] == STEREO_FRAME_WIDTH:
            left, _right = split_stereo(frame)
        else:
            left = frame

        rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return left, None

        mp_drawing.draw_landmarks(left, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark
        landmarks = {
            name: [lm[idx].x, lm[idx].y, lm[idx].z]
            for name, idx in KEYPOINTS.items()
        }
        return left, landmarks

    def close(self):
        if not self.mock:
            self.pose.close()


# MediaPipe landmark indices we care about
KEYPOINTS = {
    "left_shoulder":  mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "left_elbow":     mp_pose.PoseLandmark.LEFT_ELBOW.value,
    "right_elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "left_wrist":     mp_pose.PoseLandmark.LEFT_WRIST.value,
    "right_wrist":    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "left_hip":       mp_pose.PoseLandmark.LEFT_HIP.value,
    "right_hip":      mp_pose.PoseLandmark.RIGHT_HIP.value,
    "left_knee":      mp_pose.PoseLandmark.LEFT_KNEE.value,
    "right_knee":     mp_pose.PoseLandmark.RIGHT_KNEE.value,
    "left_ankle":     mp_pose.PoseLandmark.LEFT_ANKLE.value,
    "right_ankle":    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
}
