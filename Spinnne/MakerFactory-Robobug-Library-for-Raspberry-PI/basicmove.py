import serial
import time

# UART
PORT = "/dev/serial0"
SERIAL_CMD_BAUD = 38400
SERIAL_STD_BAUD = 115200

# Serial Protocoll
CMD_SYNC0 = 33
CMD_SYNC1 = 42
CMD_TERM  = 255

cmd_crc = 0
cmd = 0
status_byte = 0
tx_error_cnt = 0
cmd_param = [0, 0, 0, 0, 0, 0]

# Command Register (Write-Values)
CMD_REG_POWER = 5
CMD_REG_SPEED = 10
CMD_REG_GIANT_MODE = 15
CMD_REG_BALANCE_MODE = 20
CMD_REG_BODY_HEIGHT = 25
CMD_REG_TRANSLATE = 35
CMD_REG_WALK = 40
CMD_REG_ROTATE = 45
CMD_REG_DOUBLE_HEIGHT = 50
CMD_REG_DOUBLE_LENGTH = 55
CMD_REG_SINGLE_LEG_POS = 60
CMD_REG_SOUND = 65
CMD_REG_OUT1 = 70
CMD_REG_STATUS_LED = 75

# Command Register (Read-Values)
CMD_REG_SA_LEG = 100
CMD_REG_AKKU = 105
CMD_REG_PS2_ACTIVE = 110
CMD_REG_IS_WALKING = 115
CMD_REG_IS_POWER_ON = 120
CMD_REG_READ_PS2_VALUES = 125
CMD_REG_IN1 = 130

# Command Register (HW-Reset)
CMD_REG_RESET = 255

# Command Status Feedback
STATUS_ACK_OK = 64
STATUS_ERR_TERM = 1
STATUS_ERR_STATE = 2
STATUS_ERR_CRC = 3
STATUS_ERR_CMD = 255

# Move Modes
WALKMODE = 0
TRANSLATEMDOE = 1
ROTATEMODE = 2
SINGLELEGMODE = 3

# Balance Modes
BALANCEMODE_ON = 1
BALANCEMODE_OFF = 0

# Giant Modes
TRIPOD_6 = 0
TRIPOD_8 = 1
TRIPPLE_12 = 2
TRIPPLE_16 = 3
RIPPLE_12 = 4
WAVE_24 = 5

# Data States
WAIT_FOR_SYNC_0 = 0
WAIT_FOR_SYNC_1 = 1
GET_CRC = 2
GET_STATUS = 3
GET_CMD = 4
GET_PARAM0 = 5
GET_PARAM1 = 6
GET_PARAM2 = 7
GET_PARAM3 = 8
GET_PARAM4 = 9
GET_PARAM5 = 10
GET_TERM_CHAR = 11

def calc_crc(cmd, d1, d2, d3, d4):
	return CMD_SYNC0 ^ CMD_SYNC1 ^ cmd ^ d1 ^ d2 ^ d3 ^ d4

def send_data(cmd, d1, d2, d3, d4):
	crc = calc_crc(cmd, d1, d2, d3, d4)

	frame = bytes([
		CMD_SYNC0,
		CMD_SYNC1,
		crc,
		cmd,
		d1,
		d2,
		d3,
		d4,
		CMD_TERM
	])

	ser.write(frame)
	#while (not receive_ack()):
	#	ser.write(frame)

def receive_ack():
	status = False

	if (check_for_serial_data()):
		if (check_rx_crc()):
			if (status_byte == STATUS_ACK_OK):
				status = True
			else:
				status = False

	ser.reset_input_buffer()()
	return status

def check_for_serial_data():
	serial_state = WAIT_FOR_SYNC_0
	serial_timeout = 0
	temp = 0

	while (ser.in_waiting < 10):
		time.sleep(0.01)
		serial_timeout += 1

		if (serial_timeout > 50):
			break

	while (ser.in_waiting > 0):

		c_temp = ser.readline()

		if serial_state == WAIT_FOR_SYNC_0:
			if (c_temp == CMD_SYNC0):
				serial_state += 1

		elif serial_state == WAIT_FOR_SYNC_1:
			if (c_temp == CMD_SYNC1):
				serial_state += 1
			else:
				serial_state = WAIT_FOR_SYNC_0
				while (ser.in_waiting > 0):
					temp = ser.readline()
				ser.reset_input_buffer()()

		elif serial_state ==  GET_CRC:
			cmd_crc = c_temp
			serial_state += 1

		elif serial_state ==  GET_STATUS:
			status_byte = c_temp
			serial_state += 1

		elif serial_state ==  GET_CMD:
			cmd = c_temp
			serial_state += 1

		elif serial_state ==  GET_PARAM0:
			cmd_param[0] = c_temp
			serial_state += 1

		elif serial_state ==  GET_PARAM1:
			cmd_param[1] = c_temp
			serial_state += 1

		elif serial_state ==  GET_PARAM2:
			cmd_param[2] = c_temp
			serial_state += 1

		elif serial_state ==  GET_PARAM3:
			cmd_param[3] = c_temp
			serial_state += 1

		elif serial_state ==  GET_PARAM4:
			cmd_param[4] = c_temp
			serial_state += 1

		elif serial_state ==  GET_PARAM5:
			cmd_param[5] = c_temp
			serial_state += 1

		elif serial_state ==  GET_TERM_CHAR:
			if (c_temp == CMD_TERM):
				serial_state == WAIT_FOR_SYNC_0
				return True
			else:
				ser.write(STATUS_ERR_TERM)
				serial_state = WAIT_FOR_SYNC_0
				while (ser.is_waiting() > 0):
					temp = ser.readline()
				ser.reset_input_buffer()()
		
		else:
			ser.write(STATUS_ERR_STATE)
			serial_state = WAIT_FOR_SYNC_0
			while (ser.is_waiting() > 0):
				temp = ser.readline()
			ser.reset_input_buffer()()

	return False

def check_rx_crc():
	crc = CMD_SYNC0 ^ CMD_SYNC1 ^ status_byte ^ cmd ^ cmd_param[0] ^ cmd_param[1] ^ cmd_param[2] ^ cmd_param[3] ^ cmd_param[4] ^ cmd_param[5]
	if (crc == cmd_crc):
		return True
	return False

def robot_move(lateral, turn, move):
	send_data(CMD_REG_WALK, lateral, move, turn, 0)

def robot_stop():
	send_data(CMD_REG_WALK, 128, 128, 128, 0)

def robot_speed(speed):
	send_data(CMD_REG_SPEED, speed, 0, 0, 0)

def robot_height(height):
	send_data(CMD_REG_BODY_HEIGHT, height, 0, 0, 0)

def robot_giant_mode(giant_mode):
	send_data(CMD_REG_GIANT_MODE, giant_mode, 0, 0, 0)

def robot_pwr_on():
	send_data(CMD_REG_POWER, 1, 0, 0, 0)

def robot_pwr_off():
	send_data(CMD_REG_POWER, 0, 0, 0, 0)

def robot_init():
	robot_stop()
	robot_speed(100)
	robot_height(0)
	robot_giant_mode(TRIPOD_6)
	robot_pwr_off()

# UART öffnen
# UART öffnen mit der korrekten Baudrate (38400)
# Achte darauf, dass SERIAL_CMD_BAUD oben im Skript auf 38400 gesetzt ist!
with serial.Serial(PORT, SERIAL_CMD_BAUD, timeout=1) as ser:
    time.sleep(2)  # Warten bis Verbindung steht

    print("Starte Programm...")
    
    # 1. Roboter initialisieren (setzt Speed, Height, aber macht PWR OFF)
    robot_init()
    time.sleep(0.5)

    # 2. WICHTIG: Strom einschalten!
    print("Sende Power ON...")
    robot_pwr_on()
    time.sleep(1) # Kurz warten, damit Servos Strom bekommen

    # 3. Bewegung senden
    # Werte: Lateral (X), Move (Y), Turn (Rotation)
    # 128 ist meist der Stillstand/Mittelwert (0..255)
    print("Sende Bewegung...")
    
    # Test-Bewegung: Leicht vorwärts (Beispiel: Move > 128)
    # robot_move(Lateral, Turn, Move)
    robot_move(128, 128, 160) 

    time.sleep(2) # 2 Sekunden laufen lassen

    # 4. Stoppen
    print("Stoppe...")
    robot_stop()
    
    time.sleep(0.5)
    print("Erfolgreich ausgeführt!")