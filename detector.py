from __future__ import annotations
import logging
from typing import Optional

LOGGER = logging.getLogger("robot.detector")

class Detector:
    def __init__(
        self,
        enabled: bool,
        model_path: str,
        conf: float,
        save_annotated: bool,
        imgsz: int = 640,
        yolo_classes: Optional[list] = None,
        save_on: Optional[list] = None, 
    ):
        self.enabled = enabled
        self.model_path = model_path
        self.conf = conf
        self.save_annotated = save_annotated
        self.imgsz = int(imgsz)
        self.yolo_classes = yolo_classes
        self._model = None
        
        if not self.enabled: return

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._names = self._model.names
        except Exception as e:
            LOGGER.warning(f"KI konnte nicht geladen werden: {e}")
            self.enabled = False

    def process_frame(self, frame) -> Optional[tuple[bool, list[dict], Optional[object]]]:
        """
        Analysiert ein Bild und gibt gefundene Objekte zurück.
        Rückgabe: (Gefunden?, Liste_mit_Objekten, Bild_mit_Boxen)
        """
        if not self.enabled or self._model is None:
            return None
        
        try:
            # KI rechnen lassen
            results = self._model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
            if not results: return None
            
            r = results[0]
            boxes = r.boxes
            found_objects = []

            if boxes is not None:
                # Daten auslesen (CPU)
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()

                for i in range(len(xyxy)):
                    cid = int(cls[i])
                    x1, y1, x2, y2 = map(float, xyxy[i])
                    label = self._names.get(cid, str(cid))
                    
                    # Wir speichern das Objekt in einer sauberen Liste
                    found_objects.append({
                        'label': label,
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2
                    })

            # Wenn wir --show anhaben, wollen wir das Bild mit Boxen sehen
            annotated_img = None
            if self.save_annotated:
                annotated_img = r.plot()

            return (True, found_objects, annotated_img)

        except Exception as e:
            LOGGER.warning(f"Fehler bei der Erkennung: {e}")
            return None