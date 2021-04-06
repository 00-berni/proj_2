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
#	param	l	range of luminosity
#
#	return	F	N*N field matrix with sources
#	return	stars	star type array with luminosity and position of all generated stars
#!	capire come usare IMF
def initialize(N,l):
	# Definition of the empty field matrix
	# Each pixel rapresents detected luminosity of that part of sky
	F = np.zeros((N,N))
	# Setting the bin of luminosity
	bin = (l[1]-l[0])/N
	err = bin/2		# the half width of the bin
#+	Questo algoritmo va un po' rivisto
	""" 
*	Number of star in a luminosity bin function: 
		It takes the bin extremities and returns the number of star in that bin
		The number density in luminosity is n_L (L) = L**(-3) and the number of stars
		is the integral of that density in a luminosity bin 
	"""	
	n = lambda Lmax, Lmin : (Lmin**(-3) - Lmax**(-3))/2
	# Array for the numbers of stars in each bin
	Nstar = np.zeros(N,dtype=int)
	# Array for the mean luminosity for each bin
	Lstar = np.zeros(N)
	# Temporary variable for the extremities of the bins
	ltmp = [l[0], l[0] + bin]
	for i in range(0,N):
		# Calculation of the number of stars and mean luminosity in that bin
		Nstar[i] = int(n(ltmp[1],ltmp[0]))
		Lstar[i] = (ltmp[1]+ltmp[0])/2
		# Setting the new extremities
		ltmp[0]  = ltmp[1]
		ltmp[1] += bin
	# Calculation of total number of stars
	Ntot = int(Nstar.sum())
	#? Checking that Ntot is not too high for the field
	#+ Questa cosa va risolta
	if(Ntot>=N**2):
		print('.ERROR.\\ > The number of stars is too high!\\Checking the input luminosity bin')
		return None, None
	#?
	#? Controllo per me
	print(N**2)
	print(Ntot)
	print(Nstar)
	#?
	#?
	# star type array to memorize the stars values
	stars = star(np.zeros(Ntot), np.zeros(Ntot), np.zeros(Ntot))
	# Defining a temporary variable to browse through the array called stars
	cnt = 0
	for i in range(N):
		# Setting the seed of random generator to current time 
		# in order to change it everytime user runs the script
		rdm.seed(datetime.now())
		for j in range(Nstar[i]):
			# Random selection of position in field
			x = rdm.randrange(N); y = rdm.randrange(N)
			# Control condition to set the star in an empty bin
			while(F[x][y]!=0):
				x = rdm.randrange(N); y = rdm.randrange(N)
			# Setting the luminosity in the field
			F[x][y] = Lstar[i]
			# Saving the stars values in the array called stars
			stars.lum[cnt+j] = Lstar[i]
			stars.x[cnt+j]   = x
			stars.y[cnt+j]   = y
		# Updating the cnt variable
		cnt += Nstar[i]
	return F, stars

	
###* 	MAIN	*###
N = 50
l = [0.1,0.4]
F,S= initialize(N, l)
if(type(F)!=type(None)):
	s = 'Field\n'
	for i in range(N):
		for j in range(N):
			s += '%.2f'%F[i][j]
			if(j!=N-1):
				s += '\t'
		s += '\n'
	
	##* PLOTTING  ##
	from mpl_toolkits.mplot3d import Axes3D		# Function for the 3D plotting
	#
	# Setting stars positions
	x = S.x; y = S.y	
	#
	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	# Making the 2D bottom grid
	hist, xedges, yedges = np.histogram2d(x, y, bins=N, range=[[0, N], [0, N]])

	# Construct arrays for the anchor positions of the 16 bars.
	# Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
	# ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
	# with indexing='ij'.
	xpos, ypos = np.meshgrid(xedges[:-1]+0.5, yedges[:-1]+0.5)
	xpos = xpos.flatten('F')
	ypos = ypos.flatten('F')
	zpos = np.zeros_like(xpos)
	# Costruction of the array for the field
	FF = np.zeros(N*N)
	for i in range(N):
		for j in range(N):
			FF[j+N*i] = F[i][j]
	# Construct arrays 
	dx = 0.5 * np.ones_like(zpos)
	dy = dx.copy()
	dz = FF
	# Setting colors
	colors = plt.cm.jet(1.2-FF/max(FF))
	#
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')
	#
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('L')
	#
	plt.show()
