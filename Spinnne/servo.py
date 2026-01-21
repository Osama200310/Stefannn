#!/usr/bin/env python3
from __future__ import annotations
import time
import sys

# Adafruit / I²C / Servo-Bibliotheken importieren
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
except ImportError:
    print("[error] Fehlende Bibliotheken. Installiere mit:")
    print("  pip install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor")
    sys.exit(1)

# ======================
# KONSTANTEN & SETTINGS
# ======================
PAN_CHANNEL = 0
TILT_CHANNEL = 1

PAN_MIN_PULSE = 600
PAN_MAX_PULSE = 2400
TILT_MIN_PULSE = 600
TILT_MAX_PULSE = 2400

SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180

INITIAL_PAN = 82
INITIAL_TILT = 0

STEP_DEG = 5           # Schritte für Bewegung
HOLD_AT_POSITION = 20

# Neue Pan-Sequenz (abwärts)
PAN_SEQUENCE = [55]


# ======================
# SERVO-KLASSE
# ======================
class ServoController:
    def __init__(self):
        print("[info] Initialisiere I²C und PCA9685...")
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50

        self.pan = servo.Servo(
            self.pca.channels[PAN_CHANNEL],
            min_pulse=PAN_MIN_PULSE,
            max_pulse=PAN_MAX_PULSE,
        )
        self.tilt = servo.Servo(
            self.pca.channels[TILT_CHANNEL],
            min_pulse=TILT_MIN_PULSE,
            max_pulse=TILT_MAX_PULSE,
        )

        self.pan_angle = INITIAL_PAN
        self.tilt_angle = INITIAL_TILT
        self.update_servos()
        print(f"[info] Servos auf Startposition Pan={INITIAL_PAN}°, Tilt={INITIAL_TILT}°.")

    def update_servos(self):
        self.pan.angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, self.pan_angle))
        self.tilt.angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, self.tilt_angle))

    def move_pan_in_steps(self, target_angle: float):
        """Bewegt Pan in festen Schritten (STEP_DEG) zum Ziel."""
        target = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, float(target_angle)))
        current = self.pan_angle

        if current == target:
            return
    
        direction = 1 if target > current else -1
        next_angle = current
        while (direction == 1 and next_angle < target) or (direction == -1 and next_angle > target):
            next_angle += STEP_DEG * direction
            # Überlauf korrigieren
            if direction == 1 and next_angle > target:
                next_angle = target
            if direction == -1 and next_angle < target:
                next_angle = target
            self.pan_angle = next_angle
            self.update_servos()
            time.sleep(0.05)  # kleine Pause zwischen jedem 5°-Schritt
        self.pan_angle = target
        self.update_servos()

    def reset_position(self):
        """Setzt die Servos auf die initiale Position zurück."""
        self.pan_angle = INITIAL_PAN
        self.tilt_angle = INITIAL_TILT
        self.update_servos()
        print(f"[info] Position zurückgesetzt auf Pan={INITIAL_PAN}°, Tilt={INITIAL_TILT}°")

    def continuous_reset_loop(self):
        """Läuft kontinuierlich und setzt die Position zurück."""
        print("[auto] Starte kontinuierliche Positions-Reset-Schleife...")
        try:
            while True:
                self.reset_position()
                time.sleep(0.1)  # Kurze Pause zwischen den Resets
        except KeyboardInterrupt:
            print("[info] Reset-Schleife beendet.")

    def cleanup(self):
        self.pca.deinit()
        print("[info] Servos deaktiviert.")

    def pan_loop(self, sequence=PAN_SEQUENCE, hold=HOLD_AT_POSITION):
        """Fährt die Pan-Positionen nacheinander in Endlosschleife."""
        print(f"[auto] Starte Pan-Sequenz: {sequence}")
        try:
            while True:
                for target in sequence:
                    self.move_pan_in_steps(target)
                    time.sleep(hold)  # gleiche Pause an jedem Zielpunkt
        except KeyboardInterrupt:
            print("[auto] Schleife beendet (KeyboardInterrupt).")
            self.move_pan_in_steps(sequence[0])
            self.cleanup()


# ======================
# HAUPTPROGRAMM
# ======================
if __name__ == "__main__":
    controller = ServoController()
    try:
        controller.continuous_reset_loop()
    except KeyboardInterrupt:
        print("\n[info] Programm beendet.")
    finally:
        controller.cleanup()
