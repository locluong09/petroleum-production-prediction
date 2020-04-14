from random import randint
import numpy as np

def generate_point(number_of_team):
	point = [0 for i in range(number_of_team)]
	for i in range(number_of_team - 1):
		for j in range(i+1,number_of_team):
			x = randint(0,2)
			y = 2 - x
			point[i] = point[i] + x
			point[j] = point[j] + y
	return(point)
s = 0
for i in range(1000000):
	x = generate_point(8)
	x = np.sort(x)
	#if (x[4] == 11 & x[5] == 11 & x[6] ==11 &x[7] ==11):
	if (x[5] == 12 & x[6] ==12 &x[7] ==12):
		s = s+1
		print(x)

print(s)
