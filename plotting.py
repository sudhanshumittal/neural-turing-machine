import matplotlib.pyplot as plt
import numpy as np
import matplotlib
class plotting:
	def __init__(self):
		self.imageno = 1;
		np.random.seed(101)
	def draw(self, garr):
		l = len(garr)
		fig1 = plt.figure()
		for k in range(1, 1+len(garr)):
			ax1 = fig1.add_subplot(l*100+10+k)
			ax1.imshow(garr[k-1], interpolation='none')
		fig1.savefig("figures//"+str(self.imageno)+'.png')
		self.imageno += 1