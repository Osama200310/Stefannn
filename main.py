#!/usr/bin/env python3
import time
import signal
import logging
import cv2
import argparse

# --- DEINE MODULE ---
# Diese Dateien müssen im gleichen Ordner liegen!
from camera import Camera
from detector import Detector
from servo import ServoController
from Arduino_Kommunikation import ArduinoCommunication 

# Logging Setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("robot")

running = True

def handle_sigint(sig, frame):
    global running
    print("\n[STOP] Beende Roboter-Steuerung...")
    running = False

# --- EINSTELLUNGEN ---
# Zonen für die Steuerung (bei 640 Pixel Breite)
FRAME_WIDTH = 640
ZONE_LINKS = FRAME_WIDTH // 3       # ca. 213 Pixel
ZONE_RECHTS = (FRAME_WIDTH // 3) * 2 # ca. 426 Pixel


STOP_THRESHOLD = 0.40 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pfad zum KI-Modell")
    parser.add_argument("--show", action="store_true", help="Zeige Bild (nur mit Monitor/Desktop)")
    args = parser.parse_args()

    # 1. SERVO STARTEN (Kamerakopf)
    servo = None
    try:
        servo = ServoController()
        servo.reset_position() # Kamera nach vorne ausrichten
        LOGGER.info("Kamera bereit.")
    except Exception as e:
        LOGGER.warning(f"Kamera-Warnung (I2C prüfen): {e}")

    # 2. ARDUINO STARTEN
    # Wichtig: Prüfe den Port! Oft /dev/ttyACM0 oder /dev/ttyUSB0
    arduino = ArduinoCommunication(port='/dev/ttyACM0', baudrate=9600) 
    if arduino.connect():
        LOGGER.info("Arduino erfolgreich verbunden.")
    else:
        LOGGER.warning("Arduino NICHT verbunden. Testmodus aktiv.")

    # 3. KI LADEN
    # Wir nutzen imgsz=320 für mehr Geschwindigkeit auf dem Pi 4
    detector = Detector(
        enabled=True,
        model_path=args.model,
        conf=0.4,           # Nur Objekte mit >40% Sicherheit
        save_annotated=args.show,
        imgsz=640, 
        yolo_classes=None   # None = Alles erkennen. Z.B. [0] = nur Personen.
    )

    # 4. HAUPTSCHLEIFE
    with Camera(use_picam=True) as camera:
        LOGGER.info("System bereit. Starte Loop...")
        
        while running:
            # A) Bild holen
            try:
                frame = camera.capture_array()
            except Exception as e:
                LOGGER.error(f"Kamera-Fehler: {e}")
                time.sleep(0.1)
                continue

            # B) KI-Erkennung
            result = detector.process_frame(frame)
            
            # Standard-Befehl: Freie Fahrt
            command = "VORWAERTS" 

            LOGGER.info(f"Befehl: {command}")
            
            if result:
                _, detections, annotated_frame = result
                
                # Wir sortieren nach Größe (Y-Höhe), um das nächste Objekt zu finden
                detections.sort(key=lambda d: (d['y2'] - d['y1']), reverse=True)
                
                if detections:
                    obj = detections[0] # Das größte Objekt nehmen
                    
                    # Fake-3D Berechnung
                    height = obj['y2'] - obj['y1']
                    height_ratio = height / 480 # Bildhöhe ist 480
                    center_x = (obj['x1'] + obj['x2']) / 2
                    
                    # LOGIK: Ist das Hindernis nah genug (groß genug)?
                    if height_ratio > STOP_THRESHOLD:
                        if(obj['label'] == 'person'):
                            if center_x < ZONE_LINKS:
                                command = "Links"
                                LOGGER.info(f"Person LINKS -> Fahre LINKS")
                            elif center_x > ZONE_RECHTS:
                                command = "Rechts"
                                LOGGER.info(f"Person RECHTS -> Fahre RECHTS")

                    else:
                        if center_x < ZONE_LINKS:
                            command = "RECHTS"      # Hindernis links -> rechts ausweichen
                            LOGGER.info(f"HINDERNIS LINKS ({obj['label']}) -> Fahre RECHTS")
                        elif center_x > ZONE_RECHTS:
                            command = "LINKS"       # Hindernis rechts -> links ausweichen
                            LOGGER.info(f"HINDERNIS RECHTS ({obj['label']}) -> Fahre LINKS")
                        else:
                            command = "RUECKWAERTS" # Hindernis mittig -> zurück/stop
                            LOGGER.info(f"HINDERNIS MITTE ({obj['label']}) -> ZURÜCK!")
                
                # Bild anzeigen (nur mit --show)
                if args.show and annotated_frame is not None:
                    # Hilfslinien einzeichnen
                    cv2.line(annotated_frame, (ZONE_LINKS, 0), (ZONE_LINKS, 480), (0, 255, 0), 2)
                    cv2.line(annotated_frame, (ZONE_RECHTS, 0), (ZONE_RECHTS, 480), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"CMD: {command}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    cv2.imshow("Roboter Auge", annotated_frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

            # C) Befehl senden
            # Wir fügen "\n" hinzu, da Arduino 'readStringUntil' das oft braucht
            arduino.send_data(command + "\n")

            # Kleine Pause, um Überlastung zu vermeiden (ca. 20 FPS)
            # time.sleep(0.05) 

    # Aufräumen beim Beenden
    arduino.send_data("STOP\n") 
    arduino.close()
    if servo: 
        servo.cleanup()
    if args.show: 
        cv2.destroyAllWindows()
    LOGGER.info("Programm beendet.")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    main()