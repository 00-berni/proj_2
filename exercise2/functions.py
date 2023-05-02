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
import os

# dimension of the matrix
N = int(1e2+1)
# number of stars
M = int(1e2)

## Set parameters
alpha = 2
beta  = 3
# minimum and maximum masses 
m_min = 0.1; m_max = 20
# Initial Mass Function
IMF = lambda m : m**(-alpha)
IMF_min = IMF(0.1); IMF_max = IMF(20) 



class star():
    """_summary_

    :param mass: _description_
    :type mass: float
    :param lum: _description_
    :type lum: float
    :param x: _description_
    :type x: float
    :param y: _description_
    :type y: float
    """
    def __init__(self, mass: float, lum: float, x: float ,y: float):
        self.m   = mass        # star mass value
        self.lum = lum        # star luminosity value
        self.x = x        # star x coordinate
        self.y = y        # star y coordinate


def generate_mass_array(imf_min: float ,imf_max: float ,sdim: int = M) -> np.ndarray:
    """Genration of masses array with IMF distribution
    The function takes the minimum and the maximum of the
    IMF, generates a M-dimension array of random value for imf in
    [IMF_min,IMF_max] and returns a M-dimension array of masses,
    distribuited like the IMF

    :param imf_min: minimum imf value
    :type imf_min: float
    :param imf_max: maximum imf value
    :type imf_max: float
    :param sdim: number of stars, defaults to M
    :type sdim: int, optional

    :return: {dim} array of masses distributed like imf
    :rtype: np.ndarray
    """
    np.random.seed()
    return (np.random.rand(sdim)*(imf_min-imf_max)+imf_max)**(-1/alpha)

#? I have to check this function: the condition replace=False forbids M > N  
def star_location(sdim: int = M, dim: int = N) -> tuple:
    """Function to locate the stars
    It generates 2 random arrays of dimension n: 
    one is the x coordinate array and 
    y coordinate array of each star

    :param sdim: number of stars, defaults to M
    :type sdim: int, optional
    :param dim: dimension of the field, defaults to N
    :type dim: int, optional

    :return: tuple of star coordinates arrays
    :rtype: tuple
    """
    tmp = np.random.default_rng()
    X = tmp.choice(dim, size=sdim)
    Y = tmp.choice(dim, size=sdim)#, replace=False)
    return X, Y    




##* 
def update_field(F: np.ndarray, pos: tuple, lum: np.ndarray) -> np.ndarray:
    """Function to update the field
    It adds the generated stars to the field
    The shape of the field matrix is discussed
    in the next cell

    :param F: field matrix (dim,dim)
    :type F: np.ndarray
    :param pos: tuple of star coordinates arrays
    :type pos: tuple
    :param lum: luminosity array
    :type lum: np.ndarray

    :return: updated field matrix (dim,dim)
    :rtype: np.ndarray
    """
    X, Y = pos
    F[X,Y] += lum
    return F


##*   
def field_image(fig, image, F: np.ndarray, v: int = 0, sct: tuple = (0,-1)) -> None:
    """Function to represent the field
    It shows the field. It is possible to selected
    a section of the field using the parameter sct 

    :param image: matplotlib object
    :type image: Any
    :param F: field matrix
    :type F: np.ndarray
    :param v: set the color of the image, defaults to 0.
            *  1 for viridis
            *  0 for grayscale
            * -1 for inverse grayscale
    :type v: int, optional
    :param sct: selected square section of the field, defaults to [0,-1]
    :type sct: tuple, optional
    """
    a,b = sct
    # choose visualization
    if v == 0: color = 'gray'
    elif v == 1: color = 'viridis' 
    else: color = 'gray_r'
        # image
    pic = image.imshow(F[a:b,a:b], cmap=color, norm='log')
    fig.colorbar(pic, ax=image, cmap=color, norm='log', location='bottom')

def check_field(field: np.ndarray) -> np.ndarray:
    """Check the presence of negative values
    The function finds possible negative values
    and substitutes them with 0.0

    :param field: field matrix
    :type field: ndarray

    :return: checked field matrix
    :rtype: ndarray
    """
    return np.where(field < 0, 0.0, field)


##*
def initialize(dim: int = N, sdim: int = M) -> tuple:
    """Initialization function: generation of the "perfect" sky
    It generates the stars and initialized the field to make
    the sky image without any psf and noise

    :param dim: dimension of the field, defaults to N
    :type dim: int, optional
    :param sdim: number of stars, defaults to M
    :type sdim: int, optional
    :return: tuple of the field matrix F and :class: `star` object with all the informations
    :rtype: tuple
    """
    # generate a (dim,dim) matrix of 0s
    F = np.zeros([dim,dim])
    # generate masses
    m = generate_mass_array(IMF_min, IMF_max, sdim=sdim)
    # set luminosities
    L = m**beta
    # generate stars coordinates
    xs,ys = star_location(sdim, dim=dim)
    # put stars in the field
    F = check_field(update_field(F,(xs,ys),L))
    # save stars infos
    S = star(m,L,xs,ys)
    return F, S


from scipy.signal import windows
##* 
def gaussian(sigma: float = 0.5, dim: int = N) -> np.ndarray:
    """Gaussian matrix generator
    It makes a gaussian [dim,dim] matrix, centered in (0,0)

    :param sigma: the root of variance, defaults to 0.5
    :type sigma: float, optional
    :param dim: size of the field, defaults to N
    :type dim: int, optional
    
    :return: gaussian [dim,dim] matrix
    :rtype: np.ndarray
    """
    x = np.arange(dim, dtype=int)
    y = np.arange(dim, dtype=int)
    # shift to center of the field
    x -= int(dim/2);  y -= int(dim/2)
    # Gaussian function
    G = lambda r : np.exp(-r**2/sigma**2/2)
    # generate [dim,dim] matrix = G_i * G_j
    return np.outer(G(x),G(y))
    # return np.outer(windows.gaussian(N,sigma,sym=True),windows.gaussian(N,sigma,sym=True))


from scipy.signal import convolve2d

##* 
#?  Add a parameter to choose atm pfs between Gaussian and Lorentzian
def atm_seeing(f: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Atmosferic seeing function
    It convolves the field with tha Gaussian to
    make the atmosferic seeing

    :param f: field matrix
    :type f: np.ndarray
    :param sigma: the root of variance of Gaussian, defaults to 0.5
    :type sigma: float, optional
    
    :return: field matrix with seeing
    :rtype: np.ndarray
    """
    # dim of the field
    n = len(f)
    # call f_s the new field with seeing
#    f_s = f
    # take [n,n] matrix from the field
    field = f
    # convolution with gaussian
    field = fftconvolve(gaussian(sigma=sigma, dim=n), field, mode='same')
    # values are saved in each color channel 
    # to have grayscale
    return check_field(field)

##* 
def noise(n: float = 2e-4, dim: int = N) -> np.ndarray:
    """Noise generator
    It generates a (dim,dim) matrix of noise, using
    an arbitrary intensity n times
    a random value in [0,1]

    :param n: max intensity of noise, defaults to 2e-4
    :type n: float, optional
    :param dim: size of the field, defaults to N
    :type dim: int, optional

    :return: noise matrix
    :rtype: np.ndarray
    """
    np.random.seed()
    # random multiplicative (dim,dim) matrix
    N0 = np.random.random((dim, dim))*n
    return check_field(N0)

def dark_elaboration(n_value: float = 3e-4, iteration: int = 3) -> np.ndarray:
    """The function computes a number `iteration` of dark and means over them

    :param n_value: noise value, defaults to 3e-4
    :type n_value: float, optional
    :param iteration: number of dark to compute, defaults to 3
    :type iteration: int, optional

    :return: mean dark
    :rtype: np.ndarray
    """
    dark = noise(n_value)
    for i in range(iteration-1):
        dark += noise(n_value)
    dark /= iteration
    return dark


if __name__ == '__main__':
    # Current directory
    pwd = os.getcwd()
