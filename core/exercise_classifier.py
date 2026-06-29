"""
Rule-based exercise classifier.
Determines which exercise is being performed from landmark positions.
Expand CLASSIFIERS with more exercises over time.
"""

from core.pose import calculate_angle


def classify(landmarks):
    """Return best-guess exercise name or None."""
    if landmarks is None:
        return None

    for name, fn in CLASSIFIERS.items():
        if fn(landmarks):
            return name

    return None


# --- individual classifiers ---

def _is_bicep_curl(lm):
    elbow_angle = calculate_angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
    hip_angle = calculate_angle(lm["left_shoulder"], lm["left_hip"], lm["left_knee"])
    # arms moving, body relatively upright
    return elbow_angle < 130 and hip_angle > 150


def _is_squat(lm):
    knee_angle = calculate_angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
    return knee_angle < 120


def _is_shoulder_press(lm):
    elbow_angle = calculate_angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])
    # wrist above shoulder
    wrist_y = lm["left_wrist"][1]
    shoulder_y = lm["left_shoulder"][1]
    return elbow_angle > 150 and wrist_y < shoulder_y


CLASSIFIERS = {
    "squat":        _is_squat,
    "shoulder_press": _is_shoulder_press,
    "bicep_curl":   _is_bicep_curl,
}
