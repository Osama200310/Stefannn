# uart_send_test.py
import serial, time

ser = serial.Serial("/dev/serial0", 115200, timeout=1)
time.sleep(2)

while True:
    ser.write(b'\x21\x2A\x00\x00\x00\x00\x00\x00\xFF')
    print("Frame gesendet")
    time.sleep(1)
