import cv2
import numpy as np
from base_filter import BaseFilter
from pathlib import Path

class MoustacheFilter(BaseFilter):
    NOSE_TIP = 1
    
    def __init__(self, facedetector, png_path, offset_y, target):
        self.face_detector = facedetector
        self.moustache = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        self.offset_y = offset_y
        self.target = target

    def landmark_to_xy(self, frame, landmark):
        return self.face_detector.get_landmark_to_coordinates(frame, landmark)
    
    def apply(self, frame, landmarks):
        if not landmarks:
            return frame
        
        nose = landmarks[0].landmark[self.NOSE_TIP]
        # Cambiamos de landmarks a coordenadas de pixeles
        x_nose, y_nose = self.landmark_to_xy(frame, nose)
        # Obtenemos el ancho y el alto de la imagen (solo del color)
        oh, ow = self.moustache_image.shape[:2]
        # Calculamos la escala de manera dinamica para crecer el bigote segun la cara
        scale = self.target/ow
        
        new_h = max(int(oh*scale),1)
        new_w = max(int(ow*scale),1)
        
        resized = cv2.resize(self.moustache_image,((new_w, new_h)))
        
        x = x_nose - new_w // 2
        y = y_nose + self.offset_y - new_h // 2