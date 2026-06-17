import cv2
from base_filter import BaseFilter

class FiltroNariz(BaseFilter):
    def apply(self, frame, landmarks):
        if not landmarks:
            return frame
        h, w = frame.shape[:2]
        nose_lm = landmarks[0][1]  # primer rostro, índice 1 = punta de la nariz
        x = int(nose_lm.x * w)
        y = int(nose_lm.y * h)
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
        return frame
