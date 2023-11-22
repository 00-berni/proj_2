import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from .display import fast_image


##* 
class Star():
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
    def __init__(self, mass: float, lum: float, pos: tuple[np.ndarray, np.ndarray]) -> None:
        self.m   = mass       # star mass value
        self.lum = lum        # star luminosity value
        self.pos = pos        # star coordinates

    def plot_info(self, alpha: float, beta: float, sel: str = 'all') -> None:
        nstars = len(self.m)
        if sel == 'all' or sel == 'm':
            ## Plot data for masses
            plt.figure(figsize=(12,8))
            plt.title(f'Mass distribution with $\\alpha = {alpha}$ and {nstars} stars')
            bins = np.linspace(min(self.m),max(self.m),nstars//3*2)
            plt.hist(self.m,bins=bins)
            fm = bins**(-alpha)
            plt.plot(bins,fm)
            plt.xscale('log')
            plt.xlabel('m [$M_\odot$]')
            plt.ylabel('counts')
        if sel == 'all' or sel == 'L':
            ## Plot data for corrisponding luminosities
            # plot the logarithm of L
            L = np.log10(self.lum)
            plt.figure()
            plt.plot(self.m,self.lum,'.')
            plt.plot(bins,bins**beta)
            plt.yscale('log')
            plt.figure(figsize=(12,8))
            plt.title(f'Luminosity distribution with $\\beta = {beta}$ for {nstars} stars')
            bins = np.linspace(min(L),max(L),nstars//2)
            plt.hist(L,bins=bins)
            plt.xlabel('$\log{(L)}$ [$L_\odot$]')
            plt.ylabel('counts')

class Gaussian():
    def __init__(self, sigma: float, mu: float | None = None) -> None:
        self.mu = mu
        self.sigma = sigma
    
    def value(self,r: float | np.ndarray) -> float | np.ndarray:
        return np.exp(-r**2/(2*self.sigma**2))
    
    
    def kernel_norm(self, dim: int) -> float:
        if dim % 2 == 0: dim -= 1
        if self.mu is None:
            self.mu = dim // 2
        inf = -self.mu
        sup = dim - 1 - self.mu
        from scipy.integrate import quad
        return quad(self.value,inf,sup)[0]**2

    def kernel(self, dim: int) -> np.ndarray:
        if dim % 2 == 0: dim -= 1
        if self.mu is None:
            self.mu = dim // 2
        x, y = np.meshgrid(np.arange(dim),np.arange(dim))
        r = np.sqrt((x-self.mu)**2 + (y-self.mu)**2)
        sigma = self.sigma
        const = np.sqrt(2 * np.pi * sigma**2)
        kernel = self.value(r) / const
        return kernel / self.kernel_norm(dim)

    def field(self, dim: int) -> np.ndarray:
        mu = self.mu
        sigma = self.sigma
        return np.random.normal(mu,sigma,size=(dim,dim))

class Uniform():
    def __init__(self, maxval: float) -> None:
        self.max = maxval
    
    def field(self, dim: int) -> np.ndarray:
        n = self.max
        return np.random.uniform(0,n,size=(dim,dim))

## Standard values
# dimension of the field matrix
N = int(1e2)
# number of stars
M = int(1e2)
# masses range
MIN_m = 0.1
MAX_m = 20
# IMF exp
ALPHA = 2
# M-L exp
BETA = 3
# max background
BACK = 0.2/1e2
# max detector noise
NOISE = 3e-4
# sigma of PSF
SIGMA = 0.5
##

def generate_mass_array(m_min: float = MIN_m, m_max: float = MAX_m, alpha: float = ALPHA,  sdim: int = M) -> np.ndarray:
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


def star_location(sdim: int = M, dim: int = N, overlap: bool = False) -> tuple[np.ndarray,np.ndarray]:
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
    :param overlap: if `True` overlap between stars can occur, defaults to `False`
    :type overlap: bool, optional

    :return: tuple of star coordinates arrays `X` and `Y`
    :rtype: tuple
    """
    # list with all possible positions in the field
    grid = [(i,j) for i in range(dim) for j in range(dim)]
    # drawing positions from grid for stars
    ind = np.random.choice(len(grid), size=sdim, replace=overlap)
    # making arrays of coordinates
    X = np.array([grid[i][0] for i in ind])
    Y = np.array([grid[i][1] for i in ind])
    return (X, Y)    

# def update_field(F: np.ndarray, pos: tuple[np.ndarray, np.ndarray], lum: np.ndarray) -> np.ndarray:
#     """Function to update the field.
#     It adds the generated stars to the field.

#     :param F: field matrix
#     :type F: np.ndarray
#     :param pos: star coordinates
#     :type pos: tuple[np.ndarray, np.ndarray]
#     :param lum: luminosities array
#     :type lum: np.ndarray

#     :return: updated field matrix
#     :rtype: np.ndarray
#     """
#     # uppdating the field
#     F[pos] += lum
#     return F

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

def initialize(dim: int = N, sdim: int = M, masses: tuple[float, float] = (MIN_m,MAX_m), alpha: float = ALPHA, beta: float = BETA, overlap: bool = False, display_fig: bool = False) -> tuple[np.ndarray, Star]:
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
    star_pos = star_location(sdim=sdim, dim=dim, overlap=overlap)
    # updating the field matrix
    F[star_pos] += L
    # saving stars infos
    S = Star(m,L,star_pos)
    if display_fig:
        S.plot_info(alpha,beta)       
        fast_image(F,v=1,title='Inizialized Field')
    return F, S

def atm_seeing(field: np.ndarray, sigma: float = SIGMA, display_fig: bool = False) -> np.ndarray:
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
    field = np.copy(field)
    kernel = Gaussian(sigma).kernel(n)
    # convolution with gaussian seeing
    see_field = fftconvolve(field, kernel, mode='same')
    if display_fig:
        fast_image(see_field,v=1,title='Atmospheric Seeing mio')
    # see_field = gaussian_filter(field,sigma)
    # print(np.where(see_field<0))
    # if display_fig:
    #     fast_image(see_field,v=1,title='Atmospheric Seeing filter')
    # checking the field and returning it
    return see_field

def noise(distr: Uniform | Gaussian, dim: int = N, display_fig: bool = False, title: str = '') -> np.ndarray:
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
    n = distr.field(dim)
    if display_fig:
        fast_image(n,v=1,title=title)
    return n

def field_builder(dim: int = N, stnum: int = M, masses: tuple[float,float] = (MIN_m,MAX_m), star_param: tuple[float,float] = (ALPHA,BETA), atm_param: tuple[str,float | tuple] = ('Gaussian',SIGMA), back_param: tuple[str, float | tuple] = ('Uniform',BACK), det_param: tuple[str, float | tuple] = ('Uniform', NOISE), overlap: bool = False, display_fig: bool = False, results: str | None = None) -> np.ndarray | list[np.ndarray]:
    SEP = '-'*10 + '\n'
    print(SEP+f'Initialization of the field\nDimension:\t{dim} x {dim}\nNumber of stars:\t{stnum}')
    # creating the starting field
    F, S = initialize(dim,stnum,masses,*star_param,overlap=overlap,display_fig=display_fig)
    # background
    print(f'\nBackground:\t{back_param[0]} distribution')
    if back_param[0] == 'Uniform':
        n = back_param[1]
        print(f'Max background:\t{n}')
        n_b = noise(Uniform(n),dim,display_fig,title='Background')
    F_b = F + n_b
    if display_fig:
        fast_image(F_b,title='Field + Background')        
    # atm seeing
    print(f'\nAtm Seeing:\t{atm_param[0]} distribution')
    if atm_param[0] == 'Gaussian':
        sigma = atm_param[1]
        print(f'sigma:\t{sigma}')
        F_bs = atm_seeing(F_b,sigma,display_fig)
    # detector
    print(f'\nDetector noise:\t{det_param[0]} distribution')
    if det_param[0] == 'Uniform':
        n = det_param[1]
        print(f'Max detector noise:\t{n}')
        n_d = noise(Uniform(n),dim,display_fig,title='Detector noise')
    F_bsd = F_bs + n_d 
    if display_fig:
        fast_image(F_bsd,title='Field')        
    
    if results is None:
        ret_val = F_bsd
    else:    
        ret_val = [F_bsd]
        if 'F' in results or results == 'all':
            ret_val += [F]
        if 'b' in results or results == 'all':
            ret_val += [F_b]
        if 's' in results or results == 'all':
            ret_val += [F_bs]
    return ret_val    