import serial
import time

# -------------------------
# UART Einstellungen
# -------------------------
PORT = "/dev/serial0"    # UART-Port am Pi
BAUD = 38400             # Standard Robobug UART
TIMEOUT = 1              # Sekunden

# -------------------------
# Robobug Protokoll
# -------------------------
CMD_SYNC0 = 33
CMD_SYNC1 = 42
CMD_TERM = 255
CMD_REG_POWER = 5

STATUS_ACK_OK = 64

# -------------------------
# Hilfsfunktion CRC berechnen
# -------------------------
def calc_crc(cmd, d1, d2, d3, d4):
    return CMD_SYNC0 ^ CMD_SYNC1 ^ cmd ^ d1 ^ d2 ^ d3 ^ d4

# -------------------------
# Frame senden
# -------------------------
def send_power_on(ser):
    cmd, d1, d2, d3, d4 = CMD_REG_POWER, 1, 0, 0, 0
    crc = calc_crc(cmd, d1, d2, d3, d4)
    frame = bytes([CMD_SYNC0, CMD_SYNC1, crc, cmd, d1, d2, d3, d4, CMD_TERM])
    print("TX Frame:", frame)
    ser.write(frame)

# -------------------------
# UART Test
# -------------------------
def main():
    try:
        with serial.Serial(PORT, BAUD, timeout=TIMEOUT) as ser:
            time.sleep(2)  # Stabilisierung
            print("UART geöffnet, sende Power-On...")
            
            send_power_on(ser)

            # Kurze Pause, damit Robobug antworten kann
            time.sleep(0.2)

            # RX-Daten auslesen
            rx = ser.read(ser.in_waiting or 32)
            if rx:
                print("RX Daten:", rx)
                if STATUS_ACK_OK in rx:
                    print("ACK vom Robobug empfangen ✅")
                else:
                    print("Frame gesendet, aber kein ACK")
            else:
                print("Keine RX-Daten, evtl. Firmware/Mode nicht korrekt oder RX-LED blinkt nicht")

    except serial.SerialException as e:
        print("UART Fehler:", e)

if __name__ == "__main__":
    main()
