import serial
import time
serial_port='COM53'

serial_port = serial.Serial(serial_port, 9600, timeout=1)
while True:
    #print(serial_port.write('hello'.encode()))
    #print(serial_port.write(str(('1')).encode()))
    serial_port.write(b'1')
    time.sleep(2)
