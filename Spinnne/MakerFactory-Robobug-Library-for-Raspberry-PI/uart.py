import serial, time

ser = serial.Serial("/dev/serial0", 115200, timeout=1)
time.sleep(1)

ser.write(b"HELLO\n")
print("RX:", ser.read(5))
