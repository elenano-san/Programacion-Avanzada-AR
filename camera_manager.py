import cv2

class CameraManager:
    def __init__(self, camera_index: int = 0):
        """Inicializa el administrador de cámara usando la cámara por defecto."""
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.camera_index}")
    
    @staticmethod
    def list_available(max_cameras: int = 10):
        """Detecta cámaras disponibles probando hasta max_cameras índices."""
        available_cameras = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def get_frame(self):
        ret, frame = self.capture.read()
        return frame
    
    def release(self):
        self.capture.release()
        
    