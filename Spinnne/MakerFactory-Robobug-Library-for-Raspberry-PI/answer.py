import serial, time

ser = serial.Serial("/dev/serial0", 115200, timeout=1)
time.sleep(2)

ser.write(bytes(range(256)))
time.sleep(0.5)
print(ser.read(ser.in_waiting))
