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

##* star class: variable type for the star parameters
class star:
	def __init__(self, lum, x ,y):
		self.lum = lum		# star luminosity value
		self.x = x				# star x coordinate
		self.y = y				# star y coordinate

##* Function for the initialization of the fieldfrom initial mass function
#	param	N	dimension of the field
#	param	IMF	Initial Mass Function
#
#	return F	N*N field matrix with sources
#!	capire come usare IMF
def initialize(N,IMF):
	# Definition of the empty field matrix
	# Each pixel rapresents detected luminosity of that part of sky
	F = np.zeros((N,N))
	# Setting the seed of random generator to current time 
	# in order to change it everytime user runs the script
	rdm.seed(datetime.now())



##* Plotting function
def plotting():
	from import 