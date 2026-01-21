import serial, time

PORT = "/dev/serial0"
BAUD = 38400

CMD_SYNC0 = 33
CMD_SYNC1 = 42
CMD_TERM  = 255
CMD_REG_POWER = 5

def calc_crc(cmd, d1, d2, d3, d4):
    return CMD_SYNC0 ^ CMD_SYNC1 ^ cmd ^ d1 ^ d2 ^ d3 ^ d4

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# Beispiel: Power ON
cmd = CMD_REG_POWER
d1,d2,d3,d4 = 1,0,0,0
crc = calc_crc(cmd,d1,d2,d3,d4)

frame = bytes([CMD_SYNC0, CMD_SYNC1, crc, cmd, d1, d2, d3, d4, CMD_TERM])
print("TX:", frame)
ser.write(frame)