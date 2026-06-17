import sys
import os
import cv2
from camera_manager import CameraManager
from face_detector import FaceDetection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "filters"))
from filtro_nariz import FiltroNariz

if __name__ == "__main__":
    cameras = CameraManager.list_available(5)
    if not cameras:
        print("No se detectaron camaras activas.")
    else:
        print("Camaras detectadas: ")
        print(" ", " ".join(str(cam) for cam in cameras))

        choice = input("Elige el número de la camara (enter = primera): ").strip()
        selected = cameras[0] if choice == "" else int(choice)

        if selected not in cameras:
            print(f"Camara {selected} no esta en la lista.")
        else:
            camera = CameraManager(selected)
            detector = FaceDetection()
            filtro_nariz = FiltroNariz()
            WINDOW = "Vista previa de camara"
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            print(f"Camara {selected} seleccionada y abierta.")
            print("Presiona 'q' o cierra la ventana para salir.")
            try:
                while True:
                    frame = camera.get_frame()
                    if frame is None:
                        print("No se pudo leer el frame de la camara.")
                        break

                    results = detector.detect_face(frame)
                    output = filtro_nariz.apply(frame, results.face_landmarks)
                    cv2.imshow(WINDOW, output)

                    key = cv2.waitKey(1) & 0xFF
                    window_closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
                    if key == ord("q") or window_closed:
                        break
            finally:
                camera.release()
                cv2.destroyAllWindows()