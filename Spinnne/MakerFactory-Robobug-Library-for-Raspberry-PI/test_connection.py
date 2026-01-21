import serial
import time

PORT = "/dev/serial0"
BAUD = 38400

print("Lausche auf UART...")

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    while True:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting)
            print("RX:", list(data))
        time.sleep(0.2)
#Uart sda und scl