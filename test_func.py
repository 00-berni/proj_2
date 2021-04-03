"""
	SECOND EXERCISE: PSF PROJECT

	Author:	Bernardo Vettori
	Date:	
	References:
			- Shore, S.N., "The Tapestry of Modern Astrophysics"
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rdm
from datetime import datetime


#+ Provo a fare qualcosa

##* star class: variable type for the star parameters
class star(object):
	def __init__(self, lum, x ,y):
		self.lum = lum		# star luminosity value
		self.x = x		# star x coordinate
		self.y = y		# star y coordinate

##* Function for the initialization of the fieldfrom initial mass function
#	param	N	dimension of the field
#	param	n	number of stars
#	param	l	range of luminosity
#
#	return F	N*N field matrix with sources
#!	capire come usare IMF
def initialize(N,l):
	# Definition of the empty field matrix
	# Each pixel rapresents detected luminosity of that part of sky
	F = np.zeros((N,N))
	# Setting the seed of random generator to current time 
	# in order to change it everytime user runs the script
	ran = (l[1]-l[0])/N
	err = ran/2
	n = lambda Lmax, Lmin : (Lmin**(-3) - Lmax**(-3))/2
	Nstar = np.zeros(N,dtype=int)
	Lstar = np.zeros(N)
	ltmp = [l[0], l[0] + ran]
	for i in range(0,N):
		Nstar[i] = int(n(ltmp[1],ltmp[0]))
		Lstar[i] = (ltmp[1]+ltmp[0])/2
		ltmp[0]  = ltmp[1]
		ltmp[1] += ran
	Ntot = int(Nstar.sum())
	if(Ntot>=N**2):
		print('Fatal error')
		return None, None
	print(N**2)
	print(Ntot)
	print(Nstar)
	stars = star(np.zeros(Ntot), np.zeros(Ntot), np.zeros(Ntot))
	cnt = 0
	for i in range(N):
		rdm.seed(datetime.now())
		for j in range(Nstar[i]):
			x = rdm.randrange(N); y = rdm.randrange(N)
			while(F[x][y]!=0):
				x = rdm.randrange(N); y = rdm.randrange(N)
			F[x][y] = Lstar[i]
			stars.lum[cnt+j] = Lstar[i]
			stars.x[cnt+j]   = x
			stars.y[cnt+j]   = y
		cnt += Nstar[i]
	return F, stars

	

N = 50
l = [0.1,0.7]
F,S = initialize(N, l)
if(type(F)!='NoneType'):
	s = 'Field\n'
	for i in range(N):
		for j in range(N):
			s += '%f'%F[i][j]
			if(j!=N-1):
				s += '\t'
		s += '\n'

	from mpl_toolkits.mplot3d import Axes3D		# Function for the 3D plotting
	fig = plt.figure()
	ax  = fig.gca(projection='3d')

	x = np.arange(0,N,dtype=int)
	y = np.arange(0,N,dtype=int)

	for i in x:
		ax.plot(np.array([i]*N),y,F[i][y],'o')
	plt.show()