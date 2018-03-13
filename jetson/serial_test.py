from serial_io import SerialIO
import time

ard = SerialIO()

ard.start()

print(ard.read('left'))

dirs = ['left', 'right', 'middle']

while 1:
	if(type(ard.read('left')) is bytes):
		# the [0] is to get it to print as an int
		# might need struct packing for actual data
	    print('left: {}\tright: {}\tmiddle: {}'.format(
	    	ard.read('left')[0],
	    	ard.read('right')[0],
	    	ard.read('middle')[0])
	    )
	else:
		print('waiting for startup')
		time.sleep(1)

	time.sleep(0.5)
