import serial
import time
ser = serial.Serial('COM48', 9600, timeout=0,parity=serial.PARITY_EVEN, rtscts=1)

while True:
    x=ser.read()
    print(x.decode("utf-8"))
    time.sleep(1)
