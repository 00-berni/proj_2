"""
	SECOND EXERCISE: PSF PROJECT

	Author:	Bernardo Vettori
	Date:	
	References:
			- Shore, S.N., "The Tapestry of Modern Astrophysics"
"""

##* Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# dimension of the matrix
N = int(4e2+1)
# number of stars
M = int(4e2)

## Set parameters
alpha = 2
beta  = 3
# minimum and maximum masses 
m_min = 0.1; m_max = 20
# Initial Mass Function
IMF = lambda m : m**(-alpha)
IMF_min = IMF(0.1); IMF_max = IMF(20) 



##* Star class obj to collect star informations
class star(object):
#		param	mass	star mass
#		param	lum	star luminosity
# 		param	x,y 	star coordinates in the field 	
	def __init__(self,mass, lum, x ,y):
		self.m   = mass		# star mass value
		self.lum = lum		# star luminosity value
		self.x = x		# star x coordinate
		self.y = y		# star y coordinate


##* Genration of masses array with IMF distribution
#   The function takes the minimum and the maximum of the
#   IMF, generates a M-dimension array of random value for imf in
#   [IMF_min,IMF_max] and returns a M-dimension array of masses,
#   distribuited like the IMF   
#	param	min	minimum imf value
# 	param	max	maximum	imf value
#	param	sdim	number of stars. Set to M
# 
# 	return	m	{dim} array of masses distributed like imf   
def generate_mass_array(imf_min,imf_max,sdim=M):
	np.random.seed()
	return 	(np.random.rand(sdim)*(imf_min-imf_max)+imf_max)**(-1/alpha)


##* Function to locate the stars
#   It generates 2 random arrays of dimension n: 
#   one is the x coordinate array and 
#   y coordinate array of each star
#	param	sdim	number of stars. Set to M
#	param	dim	dimension of the field. Set to N
#
#
#	return	X,Y	stars coordinate arrays
#? I have to check this function: the condition replace=False forbids M > N  
def star_location(sdim=M,dim=N):
	tmp = np.random.default_rng()
	X = tmp.choice(dim, size=sdim)
	Y = tmp.choice(dim, size=sdim, replace=False)
	return X, Y	



##* Function to update the field
#   It adds the generated stars to the field
#   The shape of the field matrix is discussed
#   in the next cell
#	param	F	field matrix [dim,dim]
#	param	X,Y	coordinate arrays
#	param	l	luminosity array 
# 
# 	return	F	updated field matrix [dim,dim]
def update_field(F,X,Y,l):
	F[X,Y] = l
	return F



##* Transfer function
#?  Understand if it is reasonable 
'''   
	The implementation of this function is my choice. The function plt.imshow() takes float values in [0,1] to make grayscale image.
	I decided to convert luminosities into values in [0,1], using logaritmic values of them: 
		- I set a minimum detectable value of luminosity (called inf)
		  Luminosities < inf are represented as black pixels
		- I normaled with the maximum value of luminosity (called sup)
		  Luminosities = sup are converted in 1 (namely white)
'''
#   It converts luminosities in a scale from 0 to 1 to get grayscale
#   with function plt.imshow()
# 	param	l	array of luminosities
# 
# 	return	mag	converted magnitudes array   
def transfer_function(l):
	inf = 9e-5	# threshold to detect luminosity
	sup = 8000	# maximum luminosity
	# control the presence of 0s, change them with inf and save in an other temporary array
	tmp = np.where(l<=0,inf,l)
	# convertion
	mag = (np.log10(tmp)-np.log10(inf)) / (np.log10(sup)-np.log10(inf))
	# control the presence of negative values in mag and change them with 0
	mag = np.where(mag<0,0,mag); mag = np.where(mag>1,1,mag)
	return mag


##* Function to represent the field
#   It builts a [n,n,3] tensor (called field) and
#   converts the luminosities in [0,1] values
#   with transfer_function()   
#	param	image	image type variable
#	param	F	field matrix
#	param	v	a parameter to set the view
#			if v = -1 the image is in negative
#			set to 0 by default
#	param	sct	selected square section of the field
# 			set to [0,-1] by default
# 
# 	return	void	field is saved in image variable	
def field_image(image,F,v=0,sct=[0,-1]):
	n = len(F)
	field = np.zeros([n,n,3])
	lum = transfer_function(F)
	lum = v*(lum-1) + (1+v)*lum
	for i in range(3):
		field[:,:,i] = lum
	a,b = sct
	image.imshow(field[a:b,a:b,:])
	return 

##* Initialization function: generation of the "perfect" sky
#   It generates the stars and initialized the field to make
#   the sky image without any psf and noise
#	param	dim	dimension of the field. Set to N
#	param	sdim	number of stars. Set to M
#	
#	return	F	field matrix
#	return	S	star class obj with stars infos
def initialize(dim=N,sdim=M):
	# generate a [N,N,3] matrix of 0s
	# the third index is use for the RGB
	F = np.zeros([dim,dim])
	# generate masses
	m = generate_mass_array(IMF_min,IMF_max)
	# set luminosities
	L = m**beta
	# generate stars coordinates
	xs,ys = star_location(sdim)
	# put stars in the field
	F = update_field(F,xs,ys,L)
	# save stars infos
	S = star(m,L,xs,ys)
	return F, S

##* Gaussian matrix generator
#   It makes a gaussian [n,n] matrix, centered in [xs,ys]
#	param	sigma	the root of variance. Set to 0.9
# 	param	dim	dimension of matrix field. Set to N
#
#	return	G[n,n]	gaussian [n,n] matrix
def gaussian(sigma=0.6,dim=N):
	x = np.arange(dim,dtype=int)
	y = np.arange(dim,dtype=int)
	# shift to center of the field
	x -= int(dim/2);  y -= int(dim/2)
	# Gaussian function
	G = lambda r : np.exp(-r**2/sigma**2/2)
	# generate [n,n] matrix = G_i * G_j
	return np.outer(G(x),G(y))


##* Atmosferic seeing function
#   It convolves the field with tha Gaussian to
#   make the atmosferic seeing
#	param	f	field matrix [n,n]
#	
#	return	f_s	field matrix [n,n] with seeing
#?  Add a parameter to choose atm pfs between Gaussian and Lorentzian
def atm_seeing(f):
	# dim of the field
	n = len(f)
	# call f_s the new field with seeing
#	f_s = f
	# take [n,n] matrix from the field
	field = f
	# convolution with gaussian
	field = fftconvolve(field,gaussian(dim=n),mode='same')
	# values are saved in each color channel 
	# to have grayscale
	return field


##* Noise generator
#   It generates a [N,N,3] matrix of noise, using
#   an arbitrary value n (set to 2 * 10**-4) times
#   a random value in [0,1]
#	param	n	value of noise, set to 2e-4
#	param	dim	dimension of the field. Set to N
#
#	return	Noise	[dim,dim] matrix of noise
def noise(n = 2e-4,dim=N):
	np.random.seed()
	# random multiplicative [N,N] matrix
	N0 = np.random.random((dim,dim))*n
	return N0
