import os
import numpy as np
from scipy.signal import fftconvolve


##* 
class star():
    """Star object class.
    This class will be used only to store the 
    parameters of star object.

    :param mass: star mass
    :type mass: float
    :param lum: star luminosity
    :type lum: float
    :param pos: star coordinates (x,y)
    :type pos: tuple[np.ndarray, np.ndarray]
    """
    def __init__(self, mass: float, lum: float, pos: tuple[np.ndarray, np.ndarray]):
        self.m   = mass        # star mass value
        self.lum = lum        # star luminosity value
        self.pos = pos        # star coordinates


# dimension of the field matrix
N = int(1e2+1)
# number of stars
M = int(1e2)


# setting parameters of power laws
alpha = 2   # for IMF
beta  = 3   # for M-L relation
# minimum and maximum masses in solar mass units
m_min = 0.1; m_max = 20
# Initial Mass Function
IMF = lambda m : m**(-alpha)

# calculating IMF for the extreme masses
IMF_min = IMF(0.1); IMF_max = IMF(20) 


##* 
def generate_mass_array(m_min: float = 0.1, m_max: float = 20, alpha: float = 2,  sdim: int = M) -> np.ndarray:
    """Generating masses array from the IMF distribution
    The function takes the minimum and the maximum masses, the IMF 
    and generates a `sdim`-dimensional array of masses distributed like 
    IMF.

    The chosen method is a straightforward Monte Carlo: 
    generating uniformly random values for IMF and 
    calculating the corresponding mass.

    :param m_min: the minimum mass, defaults to 0.1 Msun
    :type m_min: float
    :param m_max: the maximum mass, defaults to 20 Msun
    :type imf_max: float
    :param alpha: the exponent of the power law, defaults to 2
    :type alpha: float
    :param sdim: number of stars, defaults to `M`
    :type sdim: int, optional

    :return: `sdim`-dimensional array of masses distributed like IMF
    :rtype: np.ndarray
    """
    # intial mass function
    IMF = lambda m : m**(-alpha)
    # evaluating IMF for the extremes
    imf_min = IMF(m_min)
    imf_max = IMF(m_max) 
    # initializing random seed
    np.random.seed()
    # generating the sample 
    return (np.random.rand(sdim)*(imf_min-imf_max)+imf_max)**(-1/alpha)

##* 
def star_location(sdim: int = M, dim: int = N) -> tuple[np.ndarray,np.ndarray]:
    """Function to locate the stars.
    It generates a list with all possible positions in the
    field matrix and draws `sdim` of those. Then it collects
    the drawn coordinates in two different arrays
    (x and y respectively).

    The parameter `replace` in `np.random.choice()` set to
    `False` forces each star has an unique position;
    in other words, no superimposition effect happens. 

    :param sdim: number of stars, defaults to `M`
    :type sdim: int, optional
    :param dim: size of the field, defaults to `N`
    :type dim: int, optional

    :return: tuple of star coordinates arrays `X` and `Y`
    :rtype: tuple

    .. todo:: 
        - #! Check the `replace` condition in `np.random.choice()`
          #? Is it a good choice?
    """
    # list with all possible positions in the field
    grid = [(i,j) for i in range(dim) for j in range(dim)]
    # drawing positions from grid for stars
    ind = np.random.choice(len(grid), size=sdim, replace=False)
    # making arrays of coordinates
    X = np.array([grid[i][0] for i in ind])
    Y = np.array([grid[i][1] for i in ind])
    return (X, Y)    

##* 
def update_field(F: np.ndarray, pos: tuple[np.ndarray, np.ndarray], lum: np.ndarray) -> np.ndarray:
    """Function to update the field.
    It adds the generated stars to the field.

    :param F: field matrix
    :type F: np.ndarray
    :param pos: star coordinates
    :type pos: tuple[np.ndarray, np.ndarray]
    :param lum: luminosities array
    :type lum: np.ndarray

    :return: updated field matrix
    :rtype: np.ndarray
    """
    # uppdating the field
    F[pos] += lum
    return F


def check_field(field: np.ndarray) -> np.ndarray:
    """Check the presence of negative values.
    The function finds possible negative values
    and substitutes them with 0.0

    :param field: field matrix
    :type field: ndarray

    :return: checked field matrix
    :rtype: ndarray
    """
    return np.where(field < 0, 0.0, field)


##*
def initialize(dim: int = N, sdim: int = M, masses: tuple[float, float] = (0.1, 20), alpha: float = 2, beta: float = 3) -> tuple[np.ndarray,star]:
    """Initialization function for the generation of the "perfect" sky
    It generates the stars and updates the field without any seeing 
    or noise effect.

    :param dim: size of the field, defaults to N
    :type dim: int, optional
    :param sdim: number of stars, defaults to M
    :type sdim: int, optional
    :param masses: the extremes of masses range, defaults to (0.1, 20)
    :type masses: tuple[float, float], optional
    :param alpha: exponent of IMF, defaults to 2
    :type alpha: float, optional
    :param beta: exponent of M-L relation, defaults to 3
    :type beta: float, optional

    :return: the field matrix F and :class: `star` object with all the stars informations
    :rtype: tuple
    """
    # generating an empty field (dim,dim) matrix
    F = np.zeros((dim,dim))
    m_inf, m_sup = masses
    # generating masses
    m = generate_mass_array(m_inf, m_sup, alpha=alpha, sdim=sdim)
    # evaluating corrisponding luminosities
    L = m**beta
    # locating the stars
    star_pos = star_location(sdim=sdim, dim=dim)
    # updating the field matrix
    F = check_field(update_field(F,star_pos,L))
    # saving stars infos
    S = star(m,L,star_pos)
    return F, S

##* 
def gaussian(sigma: float = 0.5, dim: int = N) -> np.ndarray:
    """Gaussian matrix generator
    It generates a gaussian (`dim`,`dim`) matrix, centered in 
    (`dim//2`,`dim//2`)

    :param sigma: the root of the variance, defaults to 0.5
    :type sigma: float, optional
    :param dim: size of the field, defaults to N
    :type dim: int, optional
    
    :return: gaussian (dim,dim) matrix
    :rtype: np.ndarray
    """
    # generating arrays of all positions
    x = np.arange(dim, dtype=int)
    y = np.arange(dim, dtype=int)
    # shifting to center of the field
    x -= dim // 2  
    y -= dim // 2
    # gaussian function expression
    G = lambda r : np.exp(-(r/sigma)**2/2)
    # computing the outer product
    kernel = np.outer(G(x),G(y))
    return kernel/kernel.sum()


##* 
def atm_seeing(field: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Atmosferic seeing function
    It convolves the field with tha Gaussian to
    make the atmosferic seeing

    :param field: field matrix
    :type field: np.ndarray
    :param sigma: the root of variance of Gaussian, defaults to 0.5
    :type sigma: float, optional
    
    :return: field matrix with seeing
    :rtype: np.ndarray
    """
    # dim of the field
    n = len(field)
    # coping the field in order to preserve it
    see_field = np.copy(field)
    # convolution with gaussian seeing
    see_field = fftconvolve(see_field, gaussian(sigma=sigma, dim=n), mode='same')
    # checking the field and returning it
    return check_field(see_field)

##* 
def noise(n: float, dim: int = N) -> np.ndarray:
    """Noise generator
    It generates a (dim,dim) matrix of noise, using
    an arbitrary maximum intensity n.

    :param n: max intensity of noise
    :type n: float
    :param dim: size of the field, defaults to N
    :type dim: int, optional

    :return: noise matrix
    :rtype: np.ndarray
    """
    # initializing the seed
    np.random.seed()
    # (`dim`,`dim`) matrix with random numbers 
    N0 = np.random.random((dim, dim))*n
    # checking the field
    return check_field(N0)
