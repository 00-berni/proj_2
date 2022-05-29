import numpy as np
import matplotlib.pyplot as plt
import random as rnd

'''
Funzioni che devo implementare:
	- generazione	IMF  = (m/m0)^-alpha	alpha = 2
	  distrubuzione	m/m0 = IMF^(-1/alpha)
	  m in [0.1,20]
	- luminosità	L/L0 = (m/m0)^beta	beta  = 3
	- 

'''
# dimension of the matrix
N = int(1e1)
# number of stars
M = int(2)

# Set parameters
alpha = 2
beta  = 3
m_min = 0.1; m_max = 20

IMF = lambda m : m**(-alpha)
IMF_min = IMF(0.1); IMF_max = IMF(20) 

def random_mass(min,max):
	rnd.seed()
	imf = rnd.uniform(min,max)
	return imf**(-1/alpha)

def generate_mass_array(min,max):
	m = []
	for i in range(0,M):
		m = np.append(m,random_mass(min,max))
	return m

m = generate_mass_array(IMF_min,IMF_max)

plt.plot(m)

plt.show()

def star_location(n):
	X = []; Y = []
	for i in range(0,n):
		x = rnd.randint(0,N)
		y = rnd.randint(0,N)
		while((x in X) and (y in Y)):
			print(1)
			x = rnd.randint(0,N)
			y = rnd.randint(0,N)
		X.append(x); Y.append(Y)
	X = np.array(X); Y = np.array(Y)
	return X, Y	

X, Y = star_location(M)

print(X,Y)