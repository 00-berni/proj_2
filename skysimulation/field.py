import os
import numpy as np
from numpy.typing import NDArray
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
    :type pos: tuple[NDArray, NDArray]
    """
    def __init__(self, mass: float | NDArray, lum: float | NDArray, pos: tuple[float | NDArray, float | NDArray]) -> None:
        self.m   = mass       #: star mass value
        self.lum = lum        #: star luminosity value
        self.pos = pos        #: star coordinates

    def plot_info(self, alpha: float, beta: float, sel: str = 'all') -> None:
        nstars = len(self.m)
        if sel == 'all' or sel == 'm':
            ## Plot data for masses
            plt.figure(figsize=(12,8))
            plt.title(f'Mass distribution with $\\alpha = {alpha}$ and {nstars} stars')
            # bins = np.linspace(min(self.m),max(self.m),nstars//3*2)
            cnts, bins = np.histogram(self.m,bins=nstars//3*2)
            norm = (cnts.sum() * (np.diff(bins).mean()))
            plt.stairs(cnts,bins,fill=True)
            mm = np.linspace(self.m.min(),self.m.max(),len(self.m))
            fm = lambda m: m**(-alpha)
            from scipy.integrate import quad
            imf = fm(mm)
            imf /= quad(fm,mm.min(),mm.max())[0]
            imf *= norm
            plt.plot(mm,imf,label='$IMF = m^{-\\alpha}$')
            plt.xscale('log')
            plt.xlabel('m [$M_\odot$]')
            plt.ylabel('counts')
            plt.legend()
        if sel == 'all' or sel == 'L':
            ## Plot data for corrisponding luminosities
            # plot the logarithm of L
            L = np.log10(self.lum)
            # plt.figure()
            # plt.plot(self.m,self.lum,'.')
            # plt.plot(bins,bins**beta)
            # plt.yscale('log')
            plt.figure(figsize=(12,8))
            plt.title(f'Luminosity distribution with $\\beta = {beta}$ for {nstars} stars')
            bins = np.linspace(min(L),max(L),nstars//3)
            plt.hist(L,bins=bins)
            plt.xlabel('$\log{(L)}$ [$L_\odot$]')
            plt.ylabel('counts')

class Gaussian():
    """Gaussian distribution

    """
    def __init__(self, sigma: float, mu: float | None = None) -> None:
        """Storing sigma and mean of the distribution

        :param sigma: variance
        :type sigma: float
        :param mu: mean, defaults to None
        :type mu: float | None, optional
        """
        # mean
        self.mu = mu
        # sigma
        self.sigma = sigma
    
    def info(self) -> None:
        """Printing the information about the distribution
        """
        print('Gaussian distribution')
        print(f'mean:\t{self.mu}')
        print(f'sigma:\t{self.sigma}')

    def value(self,r: float | NDArray) -> float | NDArray:
        """Computing the value

        :param r: variable
        :type r: float | NDArray
        
        :return: the value of the distribution
        :rtype: float | NDArray
        """
        x = r/self.sigma
        return np.exp(-x**2/2)
    
    def kernel_norm(self, dim: int) -> float:
        """Computing the normalization coefficient of a matrix

        :param dim: size of the field
        :type dim: int
        
        :return: normalization coefficient
        :rtype: float
        """
        # kernel must have a odd size
        if dim % 2 == 0: dim -= 1
        if self.mu is None:
            self.mu = dim // 2
        # edges of integration
        inf = -self.mu
        sup = dim - 1 - self.mu
        from scipy.integrate import quad
        return quad(self.value,inf,sup)[0]

    def kernel(self, dim: int) -> NDArray:
        """Computing a Gaussian kernel

        :param dim: size of the field
        :type dim: int
        
        :return: kernel
        :rtype: NDArray
        """
        # kernel must have a odd size
        if dim % 2 == 0: dim -= 1
        if self.mu is None:
            self.mu = dim // 2
        # generating coordinates
        x, y = np.meshgrid(np.arange(dim),np.arange(dim))
        # computing the distance from the center
        r = np.sqrt((x-self.mu)**2 + (y-self.mu)**2)
        # computing kernel
        kernel = self.value(r)
        return kernel / self.kernel_norm(dim)

    def field(self, dim: int) -> NDArray:
        """Drawing values from Gaussian distribution

        :param dim: size of the field
        :type dim: int
        
        :return: matrix
        :rtype: NDArray
        """
        mu = self.mu
        sigma = self.sigma
        rng = np.random.default_rng()
        return rng.normal(mu,sigma,size=(dim,dim))

class Uniform():
    """Uniform distribution
    """
    def __init__(self, maxval: float, minval: float = 0) -> None:
        """Storing parameters of uniform distribution

        :param maxval: maximum value
        :type maxval: float
        :param minval: minimum value, defaults to 0
        :type minval: float, optional
        """
        self.max = maxval
        self.min = minval

    def info(self) -> None:
        print('Uniform distribution')
        print(f'minval:\t{self.min}')
        print(f'maxval:\t{self.max}')

    def field(self, dim: int) -> NDArray:
        n = self.max
        rng = np.random.default_rng()
        return rng.uniform(self.min,n,size=(dim,dim))

class Poisson():
    def __init__(self,lam: float, k: float = 1) -> None:
        self.lam = lam
        self.k = k

    def info(self) -> None:
        print('Poisson distribution')
        print(f'lambda:\t{self.lam}')

    def field(self, dim: int) -> NDArray:
        rng = np.random.default_rng()
        return self.k * rng.poisson(self.lam,size=(dim,dim))
        


def from_parms_to_distr(params: tuple[str, float | tuple], infos: bool = False) -> Gaussian | Uniform:
    name, vals = params
    if name == 'Gaussian' or name == 'Normal':
        mu, sigma = vals
        distr = Gaussian(sigma,mu)
    elif name == 'Uniform':
        max_val = vals
        distr = Uniform(max_val)
    elif name == 'Poisson':
        lam, k = vals
        distr = Poisson(lam,k)
    if infos:
        distr.info()
    return distr

## Standard values
MSOL = 1.989e+33 # g
LSOL = 3.84e+33 # erg/s
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
# normalization constant
K = 1/(MAX_m**BETA)
# mean background value
BACK_MEAN = MAX_m**BETA * 0.01e-2
BACK_SIGMA = BACK_MEAN * 20e-2
BACK_PARAM = ('Gaussian',(BACK_MEAN, BACK_SIGMA))
# mean detector noise
NOISE_MEAN = 1e-1
NOISE_SIGMA = NOISE_MEAN * 50e-2
NOISE_PARAM = ('Gaussian',(NOISE_MEAN,NOISE_SIGMA))
# sigma of seeing
SEEING_SIGMA = 3       
ATM_PARAM = ('Gaussian',SEEING_SIGMA)                                                                                     
##

def generate_mass_array(m_min: float = MIN_m, m_max: float = MAX_m, alpha: float = ALPHA,  sdim: int = M) -> NDArray:
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
    :rtype: NDArray
    """
    # intial mass function
    IMF = lambda m : m**(-alpha)
    # evaluating IMF for the extremes
    imf_min = IMF(m_min)
    imf_max = IMF(m_max) 
    # generating the sample 
    rng = np.random.default_rng()
    return rng.uniform(imf_max,imf_min,sdim)**(-1/alpha)


def star_location(sdim: int = M, dim: int = N, overlap: bool = False) -> tuple[NDArray,NDArray]:
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
    rng = np.random.default_rng()
    ind = rng.choice(len(grid), size=sdim, replace=overlap)
    # making arrays of coordinates
    X = np.array([grid[i][0] for i in ind])
    Y = np.array([grid[i][1] for i in ind])
    return (X, Y)    

# def update_field(F: NDArray, pos: tuple[NDArray, NDArray], lum: NDArray) -> NDArray:
#     """Function to update the field.
#     It adds the generated stars to the field.

#     :param F: field matrix
#     :type F: NDArray
#     :param pos: star coordinates
#     :type pos: tuple[NDArray, NDArray]
#     :param lum: luminosities array
#     :type lum: NDArray

#     :return: updated field matrix
#     :rtype: NDArray
#     """
#     # uppdating the field
#     F[pos] += lum
#     return F

def initialize(dim: int = N, sdim: int = M, masses: tuple[float, float] = (MIN_m,MAX_m), alpha: float = ALPHA, beta: float = BETA, overlap: bool = False, display_fig: bool = False, **kwargs) -> tuple[NDArray, Star]:
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
    L = m**beta * K 
    # locating the stars
    star_pos = star_location(sdim=sdim, dim=dim, overlap=overlap)
    # updating the field matrix
    F[star_pos] += L #/max(L)
    # saving stars infos
    S = Star(m,L,star_pos)
    if display_fig:
        S.plot_info(alpha,beta)       
        if 'title' not in kwargs:
            kwargs['title'] = 'Inizialized Field'
        fast_image(F,**kwargs)
    return F, S

def atm_seeing(field: NDArray, sigma: float = SEEING_SIGMA,ext_val: int = 7, display_fig: bool = False, **kwargs) -> NDArray:
    """Atmosferic seeing function
    It convolves the field with tha Gaussian to
    make the atmosferic seeing

    :param field: field matrix
    :type field: NDArray
    :param sigma: the root of variance of Gaussian, defaults to 0.5
    :type sigma: float, optional
    
    :return: field matrix with seeing
    :rtype: NDArray
    """
    # dim of the field
    n = len(field)
    tmp_field = np.zeros((n+2*ext_val, n+2*ext_val))
    # coping the field in order to preserve it
    cut = slice(ext_val, n+ext_val)
    tmp_field[cut,cut] += field
    from astropy.convolution import convolve_fft, Gaussian2DKernel
    see_field = convolve_fft(tmp_field, Gaussian2DKernel(sigma))
    if display_fig:
        if 'title' not in kwargs:
            kwargs['title'] = 'Atmospheric Seeing '
        fast_image(see_field,**kwargs)
        fast_image(see_field[cut,cut],**kwargs)
    # checking the field and returning it
    return see_field[cut,cut]

def noise(params: tuple[str, float | tuple], dim: int = N, infos: bool = False, display_fig: bool = False, **kwargs) -> NDArray:
    """Noise generator
    It generates a (dim,dim) matrix of noise, using
    an arbitrary maximum intensity n.

    :param n: max intensity of noise
    :type n: float
    :param dim: size of the field, defaults to N
    :type dim: int, optional

    :return: noise matrix
    :rtype: NDArray
    """
    distr = from_parms_to_distr(params,infos)
    n = distr.field(dim)
    if display_fig:
        fast_image(n,**kwargs)
        plt.figure()
        plt.title('Distribution')
        plt.hist(n.flatten(),len(n.flatten())//100)
        plt.show()
    if len(np.where(n < 0)) != 0:
        n = np.sqrt(n**2)
    return n * K

def field_builder(dim: int = N, stnum: int = M, masses: tuple[float,float] = (MIN_m,MAX_m), star_param: tuple[float,float] = (ALPHA,BETA), atm_param: tuple[str,float | tuple] = ATM_PARAM, back_param: tuple[str, float | tuple] = BACK_PARAM, det_param: tuple[str, float | tuple] = NOISE_PARAM, overlap: bool = False, results: str | None = None, display_fig: bool = False, **kwargs) -> list[Star | NDArray]:
    """Constructor of the field

    :param dim: size of the field, defaults to N
    :type dim: int, optional
    :param stnum: number of starfish, defaults to M
    :type stnum: int, optional
    :param masses: mass range extrema, defaults to (MIN_m,MAX_m)
    :type masses: tuple[float,float], optional
    :param star_param: exponents, defaults to (ALPHA,BETA)
    :type star_param: tuple[float,float], optional
    :param atm_param: seeing, defaults to ATM_PARAM
    :type atm_param: tuple[str,float  |  tuple], optional
    :param back_param: background, defaults to BACK_PARAM
    :type back_param: tuple[str, float  |  tuple], optional
    :param det_param: detector, defaults to NOISE_PARAM
    :type det_param: tuple[str, float  |  tuple], optional
    :param overlap: if `True` stars can have the same position, defaults to False
    :type overlap: bool, optional
    :param results: chosen results to return, defaults to None
    :type results: str | None, optional
    :param display_fig: if `True` pictures are shown, defaults to False
    :type display_fig: bool, optional
    
    :return: stars and field (and additional results)
    :rtype: list[NDArray]
    """
    SEP = '-'*10 + '\n'
    print(SEP+f'Initialization of the field\nDimension:\t{dim} x {dim}\nNumber of stars:\t{stnum}')
    # creating the starting field
    F, S = initialize(dim,stnum,masses,*star_param,overlap=overlap,display_fig=display_fig,**kwargs)
    # background
    print('\nBackground:')
    kwargs['title'] = 'Background'
    n_b = noise(back_param,dim,infos=True,display_fig=display_fig,**kwargs)
    F_b = F + n_b
    if display_fig:
        kwargs['title'] = 'Field + Background'
        fast_image(F_b,**kwargs)        
    # atm seeing
    print(f'\nAtm Seeing:\n{atm_param[0]} distribution')
    if atm_param[0] == 'Gaussian':
        sigma = atm_param[1]
        print(f'sigma:\t{sigma}')
        kwargs['title'] = 'Atmospheric Seeing'
        F_bs = atm_seeing(F_b,sigma,display_fig=display_fig,**kwargs)
    # detector
    print('\nDetector noise:')
    kwargs['title'] = 'Detector noise'
    n_d = noise(det_param,dim,infos=True,display_fig=display_fig,**kwargs)
    F_bsd = F_bs + n_d 
    # if display_fig:
    kwargs['title'] = 'Final Field'
    fast_image(F_bsd,**kwargs)        
    
    ret_val = [S, F_bsd]
    if results is not None:
        if 'F' in results or results == 'all':
            ret_val += [F]
        if 'b' in results or results == 'all':
            ret_val += [F_b]
        if 's' in results or results == 'all':
            ret_val += [F_bs]
    return ret_val    