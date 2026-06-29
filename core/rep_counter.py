from core.pose import calculate_angle


class RepCounter:
    """
    Counts reps by watching a joint angle cross a threshold in both directions.
    down_threshold: angle below which we consider the joint "contracted"
    up_threshold:   angle above which we consider the joint "extended"
    """

    def __init__(self, down_threshold=70, up_threshold=160):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.reps = 0
        self._stage = None  # "up" or "down"

    def update(self, angle):
        if angle is None:
            return self.reps

        if angle > self.up_threshold:
            self._stage = "up"
        if angle < self.down_threshold and self._stage == "up":
            self._stage = "down"
            self.reps += 1

        return self.reps

    def reset(self):
        self.reps = 0
        self._stage = None


def get_angle_for_exercise(exercise, landmarks):
    """Return the primary joint angle for a given exercise."""
    if landmarks is None:
        return None

    handlers = {
        "bicep_curl":      _angle_elbow_left,
        "squat":           _angle_knee_left,
        "shoulder_press":  _angle_elbow_left,
        "tricep_pushdown": _angle_elbow_left,
    }

    handler = handlers.get(exercise)
    return handler(landmarks) if handler else None


def _angle_elbow_left(lm):
    return calculate_angle(lm["left_shoulder"], lm["left_elbow"], lm["left_wrist"])


def _angle_knee_left(lm):
    return calculate_angle(lm["left_hip"], lm["left_knee"], lm["left_ankle"])
