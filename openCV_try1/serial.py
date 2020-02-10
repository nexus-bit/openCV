import serial
print('serial ' + serial.__version__)

# Set a PORT Number & baud rate
PORT = 'COM7'
BaudRate = 9600

ARD= serial.Serial(PORT,BaudRate)

A=1234
B=5678

A=str(A)
B=str(B)
Trans="Q" + A + B
Trans= Trans.encode('utf-8')


while (True):
    ARD.write(Trans)  # Q12345678 전송

