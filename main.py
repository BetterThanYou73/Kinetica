"""
Kinetica — entry point
Run with: python main.py
Run in mock mode (no camera): python main.py --mock
"""

import argparse
import cv2

from core.pose import PoseDetector
from core.rep_counter import RepCounter, get_angle_for_exercise
from core.exercise_classifier import classify


def run(mock=False):
    detector = PoseDetector(mock=mock, stereo=not mock)
    counter = RepCounter()
    current_exercise = None

    if not mock:
        cap = cv2.VideoCapture(0)
        # stereo USB camera needs explicit resolution — defaults may be lower
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError("Camera not found. Check 'ls /dev/video*' and adjust VideoCapture index.")
    else:
        cap = None

    print("Kinetica started. Press 'q' to quit.")

    try:
        while True:
            if mock:
                # blank frame for UI testing without camera
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype="uint8")
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame, landmarks = detector.process(frame)

            detected = classify(landmarks)
            if detected and detected != current_exercise:
                current_exercise = detected
                counter.reset()
                print(f"Exercise detected: {current_exercise}")

            angle = get_angle_for_exercise(current_exercise, landmarks)
            reps = counter.update(angle)

            # overlay
            label = f"{current_exercise or 'detecting...'} | reps: {reps}"
            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Kinetica", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run without camera")
    args = parser.parse_args()
    run(mock=args.mock)
