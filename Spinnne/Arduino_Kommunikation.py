import serial
import time

class ArduinoCommunication:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
    
    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Warten auf Arduino-Reset
            return True
        except Exception as e:
            print(f"Verbindungsfehler: {e}")
            return False
    
    def send_data(self, data):
        if self.serial and self.serial.is_open:
            try:
                # WICHTIG: Hier wird in Bytes kodiert
                self.serial.write(str(data).encode('utf-8'))
                return True
            except Exception as e:
                print(f"Sendefehler: {e}")
                return False
        return False
    
    def receive_data(self):
        if self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting > 0:
                    return self.serial.readline().decode('utf-8').strip()
            except Exception as e:
                print(f"Empfangsfehler: {e}")
        return None
    
    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()