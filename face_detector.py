import os
import cv2
from mediapipe.tasks.python.core import base_options as base_options_lib
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat

#mediapipe
#--tasks
#----vision
#--------FaceLandmarker

_WHITE = (255, 255, 255)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


class FaceDetection:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        running_mode = (
            running_mode_lib.VisionTaskRunningMode.IMAGE
            if static_image_mode
            else running_mode_lib.VisionTaskRunningMode.VIDEO
        )
        options = FaceLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = FaceLandmarker.create_from_options(options)
        self._frame_ts = 0
        self._static = static_image_mode

    def detect_face(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        if self._static:
            return self.detector.detect(mp_image)
        self._frame_ts += 1
        return self.detector.detect_for_video(mp_image, self._frame_ts)

    def draw_faces(self, image):
        results = self.detect_face(image)
        out_img = image.copy()

        if not results.face_landmarks:
            return out_img

        h, w = image.shape[:2]

        for face_landmarks in results.face_landmarks:
            coords = self.get_landmark_to_coordinates(image, face_landmarks)

            # Triangulación de Delaunay sobre los landmarks para generar la malla
            subdiv = cv2.Subdiv2D((0, 0, w, h))
            for pt in coords:
                x, y = pt
                if 0 <= x < w and 0 <= y < h:
                    subdiv.insert((float(x), float(y)))

            for t in subdiv.getTriangleList().astype(int):
                pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
                if all(0 <= p[0] < w and 0 <= p[1] < h for p in pts):
                    cv2.line(out_img, pts[0], pts[1], _WHITE, 1)
                    cv2.line(out_img, pts[1], pts[2], _WHITE, 1)
                    cv2.line(out_img, pts[2], pts[0], _WHITE, 1)

        return out_img

    def get_landmark_to_coordinates(self, image, face_landmarks):
        h, w = image.shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
