"""
	SECOND EXERCISE: PSF PROJECT

	Author:	Bernardo Vettori
	Date:	
	References:
			- Shore, S.N., "The Tapestry of Modern Astrophysics"
"""

import numpy as np
import random as rdm
from datetime import datetime

##* Function for the initialization of the fieldfrom initial mass function
#	param	N	dimension of the field
#	param	IMF	Initial Mass Function
#
#	return F	N*N field matrix with sources
def initialize(N,IMF):
	# Definition of the empty field matrix
	# Each pixel rapresents detected luminosity of that part of sky
	F = np.zeros((N,N))
	# Setting the seed of random generator to current time 
	# in order to change it everytime user runs the script
	rdm.seed(datetime.now())

