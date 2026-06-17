import cv2
import numpy as np
from base_filter import BaseFilter
from pathlib import Path

_BIGOTE_PATH = Path(__file__).parent.parent / "assets" / "bigote.png"

class MoustacheFilter(BaseFilter):
    NOSE_TIP = 1

    def __init__(self, png_path=_BIGOTE_PATH, offset_y=20, target=100):
        self.moustache = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        self.offset_y = offset_y
        self.target = target

    def apply(self, frame, landmarks):
        if not landmarks:
            return frame

        h, w = frame.shape[:2]
        nose = landmarks[0][self.NOSE_TIP]
        # Cambiamos de landmarks a coordenadas de pixeles
        x_nose = int(nose.x * w)
        y_nose = int(nose.y * h)

        # Obtenemos el ancho y el alto de la imagen (solo del color)
        oh, ow = self.moustache.shape[:2]
        # Calculamos la escala de manera dinamica para crecer el bigote segun la cara
        scale = self.target / ow

        # Cambiamos el alto y el ancho
        new_h = max(int(oh * scale), 1)
        new_w = max(int(ow * scale), 1)

        # Reescalamos la imagen del bigote
        resized = cv2.resize(self.moustache, (new_w, new_h))

        x = x_nose - new_w // 2
        y = y_nose + self.offset_y - new_h // 2
        self.overlay_rgba_on_rgb(frame, resized, x, y)
        return frame

    def overlay_rgba_on_rgb(self, frame, moustache, x, y):
        # Obtener el alto y el ancho del frame principal
        fh, fw = frame.shape[:2]

        # Obtener el alto y ancho del bigote
        mh, mw = moustache.shape[:2]

        # Calcular los límites reales del frame
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + mw)
        y2 = min(fh, y + mh)

        if x1 >= x2 or y1 >= y2:
            return

        # Region Of Interest
        roi = frame[y1:y2, x1:x2]

        # Extraer parte visible
        rx1 = x1 - x
        ry1 = y1 - y
        rx2 = rx1 + (x2 - x1)
        ry2 = ry1 + (y2 - y1)

        crop = moustache[ry1:ry2, rx1:rx2]

        # Separamos la región RGB de A en RGBA, para que ignore la parte de la transparencia (A) y se quede todo lo que tiene color (RGB)
        rgb = crop[..., :3]

        # Extraemos canal Alpha
        alpha = crop[..., 3:4] / 255

        # Combinar Overlay
        blended = alpha * rgb + (1.0 - alpha) * roi

        roi[:] = blended.astype(np.uint8)
