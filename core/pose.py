import cv2
import numpy as np
import urllib.request
import os

# Stereo camera outputs a side-by-side 2560x720 frame.
# We use the left half for pose tracking (right half available for depth later).
STEREO_FRAME_WIDTH = 2560
SINGLE_WIDTH = STEREO_FRAME_WIDTH // 2  # 1280

# MoveNet Thunder keypoint indices
KEYPOINTS = {
    "nose":           0,
    "left_eye":       1,
    "right_eye":      2,
    "left_ear":       3,
    "right_ear":      4,
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_elbow":     7,
    "right_elbow":    8,
    "left_wrist":     9,
    "right_wrist":    10,
    "left_hip":       11,
    "right_hip":      12,
    "left_knee":      13,
    "right_knee":     14,
    "left_ankle":     15,
    "right_ankle":    16,
}

# Skeleton connections for drawing
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "movenet_thunder.tflite")
MODEL_URL = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
MODEL_INPUT_SIZE = 256  # MoveNet Thunder expects 256x256
CONFIDENCE_THRESHOLD = 0.3


def _download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MoveNet Thunder model (~12MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


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
        self._interpreter = None

        if not mock:
            _download_model()
            self._load_model()

    def _load_model(self):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow as tf
            tflite = tf.lite

        self._interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self._interpreter.allocate_tensors()
        self._input  = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

    def process(self, frame):
        """Return (annotated_left_frame, landmarks_dict) or (frame, None) on failure."""
        if self.mock:
            return frame, None

        if self.stereo and frame.shape[1] == STEREO_FRAME_WIDTH:
            left, _right = split_stereo(frame)
        else:
            left = frame

        h, w = left.shape[:2]

        # preprocess
        img = cv2.resize(left, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(img.astype(np.float32), axis=0)

        self._interpreter.set_tensor(self._input["index"], tensor)
        self._interpreter.invoke()

        # output shape: [1, 1, 17, 3]  →  [y, x, confidence] per keypoint
        raw = self._interpreter.get_tensor(self._output["index"])[0][0]

        landmarks = {}
        for name, idx in KEYPOINTS.items():
            y, x, conf = raw[idx]
            if conf >= CONFIDENCE_THRESHOLD:
                landmarks[name] = [float(x), float(y), float(conf)]

        if not landmarks:
            return left, None

        self._draw(left, landmarks, w, h)
        return left, landmarks

    def _draw(self, frame, landmarks, w, h):
        for name, (x, y, _) in landmarks.items():
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        for a, b in CONNECTIONS:
            if a in landmarks and b in landmarks:
                x1, y1, _ = landmarks[a]
                x2, y2, _ = landmarks[b]
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame, pt1, pt2, (0, 200, 255), 2)

    def close(self):
        pass
