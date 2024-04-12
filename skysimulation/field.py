import os
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from .display import fast_image, field_image


### CLASS DEFINITIONS
class Star():
    """Star object class.
    This class will be used only to store the 
    parameters of star object.

    Attributes
    ----------
    mass : float | NDArray
        star mass
    lum : float | NDArray
        star luminosity
    pos : tuple[float  |  NDArray, float  |  NDArray]
        star coordinates (x,y)
    """
    def __init__(self, mass: float | NDArray, lum: float | NDArray, pos: tuple[float | NDArray, float | NDArray]) -> None:
        """Constructor of the class

        Parameters
        ----------
        mass : float | NDArray
            star mass
        lum : float | NDArray
            star luminosity
        pos : tuple[float  |  NDArray, float  |  NDArray]
            star coordinates (x,y)
        """
        self.m   = mass       #: star mass value
        self.lum = lum        #: star brightness value
        self.pos = pos        #: star coordinates

    def plot_info(self, alpha: float, beta: float, sel: {'all', 'm', 'L'} = 'all') -> None:
        """To plot the mass and brightness distribution of the stars sample

        Through the `sel` parameter it is possible to choose to plot:

            * `'m'`     : mass distribution of the sample
            * `'L'`     : brightness distribution of the sample
            * `'all'`   : both  

        Parameters
        ----------
        alpha : float
            exponent of the IMF
        beta : float
            exponent of the M-L relation
        sel : {'all', 'm', 'L'}, optional
            selection parameter, by default `'all'`
        """
        nstars = len(self.m)    #: number of stars
        ## Mass Distribution Plot
        if sel == 'all' or sel == 'm':
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
        ## Brightness Distribution Plot
        if sel == 'all' or sel == 'L':
            # compute the logarithm of the values
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

    Attributes
    ----------
    mu : float | None
        the mean
    sigma : float
        root of the variance
    """
    def __init__(self, sigma: float, mu: float | None = None) -> None:
        """Storing sigma and mean of the distribution

        Parameters
        ----------
        sigma : float
            root of the variance
        mu : float | None, optional
            the mean, by default `None` 
        """
        self.mu = mu            #: the mean 
        self.sigma = sigma      #: the root of the variance
    
    def mean(self) -> float:
        return self.mu

    def info(self) -> None:
        """Printing the information about the distribution
        """
        print('Gaussian distribution')
        print(f'mean:\t{self.mu}')
        print(f'sigma:\t{self.sigma}')

    def value(self,r: float | NDArray) -> float | NDArray:
        """To compute the value of the distribution

        Parameters
        ----------
        r : float | NDArray
            variable value

        Returns
        -------
        float | NDArray
            the value of the distribution in `r`
        """
        x = r/self.sigma
        return np.exp(-x**2/2)
##? -- -- -- -- -- -- -- -- -- -- ?##    
    def kernel_norm(self, dim: int) -> float:
        """To compute the normalization coefficient

        Kernel has to be normalized

        Parameters
        ----------
        dim : int
            size of the field

        Returns
        -------
        float
            normalization coefficient
        
        Notes
        -----
        The normalization coefficient is defined as:

        .. math::  2\pi \, \int_{-\mu}^{\mu} \exp{()}\, dr 
        
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

    def kernel(self) -> NDArray:
        """Computing a Gaussian kernel

        :param dim: size of the field
        :type dim: int
        
        :return: kernel
        :rtype: NDArray
        """
        # kernel must have a odd size
        dim = 8*self.sigma + 1
        if self.mu is None:
            self.mu = dim // 2
        # generating coordinates
        x, y = np.meshgrid(np.arange(dim),np.arange(dim))
        # computing the distance from the center
        r = np.sqrt((x-self.mu)**2 + (y-self.mu)**2)
        # computing kernel
        kernel = self.value(r)
        return kernel / kernel.sum()
##? -- -- -- -- -- -- -- -- -- -- ?##    
    def field(self, shape: int | Sequence[int], seed: int | None = None) -> NDArray:
        """To compute a matrix of Gaussian distributed values

        Function draws values from a Gaussian distribution
        defined by the class attributes `self.mu` and 
        `self.sigma`

        It is possible to pass a seed value for the 
        random generator
        
        Parameters
        ----------
        dim : int
            size of the field
        seed : int | None, optional
            seed for the random generator, by default None

        Returns
        -------
        NDArray
            matrix of gaussian distributed values
        
        See also
        --------
        numpy.random.default_rng : Construct a new Generator with the default BitGenerator
        """
        mu = self.mu
        sigma = self.sigma
        # construct a random generator
        rng = np.random.default_rng(seed=seed)
        # draw values from normal distribution
        return rng.normal(mu,sigma,size=shape)

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

    def mean(self) -> float:
        return (self.max - self.min)/2

    def info(self) -> None:
        print('Uniform distribution')
        print(f'minval:\t{self.min}')
        print(f'maxval:\t{self.max}')

    def field(self, shape: int | Sequence[int], seed: int | None = None) -> NDArray:
        n = self.max
        rng = np.random.default_rng(seed=seed)
        return rng.uniform(self.min,n,size=shape)

class Poisson():
    def __init__(self,lam: float, k: float = 1) -> None:
        self.lam = lam
        self.k = k

    def mean(self) -> float:
        return self.lam

    def info(self) -> None:
        print('Poisson distribution')
        print(f'lambda:\t{self.lam}')

    def field(self, shape: int | Sequence[int], seed: int | None = None) -> NDArray:
        rng = np.random.default_rng(seed = seed)
        return self.k * rng.poisson(self.lam,size=shape)
        
DISTR = Gaussian | Uniform | Poisson

def from_parms_to_distr(params: tuple[str, float] | tuple[str, tuple], infos: bool = False) -> Gaussian | Uniform:
    """To get from input parameter the chosen distribution

    Parameters
    ----------
    params : tuple[str, float] | tuple[str, tuple]
        parameters of the distribution
    infos : bool, optional
        if `True` the information about 
        the distribution is printed, by default `False`

    Returns
    -------
    Gaussian | Uniform
        the chosen distribution class object
    """
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

def field_convolve(field: NDArray, kernel: NDArray, bkg: DISTR, norm_cost: float = 1) -> NDArray:
    """To convolve the field with a kernel

    Parameters
    ----------
    field : NDArray
        field matrix
    kernel : NDArray
        kernel to provide convolution
    bkg : DISTR
        background distribution

    Returns
    -------
    conv_field : NDArray
        field convolved with the kernel
    
    See also
    --------
    scipy.signal.convolve2d : 2-D convolution

    Notes
    -----

    """
    from scipy.signal import convolve2d    
    dim = len(field)                        #: size of the field
    pad_size = (len(kernel)-1) // 2         #: number of pixels to pad the field
    pad_dim = dim + 2*pad_size              #: new size of the board
    pad_slice = slice(pad_size,-pad_size)   #: field size cut
    # start from a board with only background 
    tmp_field = bkg.field(shape=(pad_dim,pad_dim)) * norm_cost
    # put the field
    tmp_field[pad_slice,pad_slice] = field.copy()
    # convolve
    conv_field = convolve2d(tmp_field, kernel, mode='same',boundary='fill',fillvalue=bkg.mean()*norm_cost)
    # cut the field
    return conv_field[pad_slice,pad_slice]



### STANDARD VALUES
# MSOL = 1.989e+33 # g
# LSOL = 3.84e+33 # erg/s
N = int(1e2)            #: size of the field matrix
M = int(1e2)            #: number of stars
MIN_m = 0.5             #: min mass value for stars
MAX_m = 10              #: max mass value for stars
ALPHA = 2               #: IMF exp
BETA = 3                #: M-L exp
K = 1/(MAX_m**BETA)     #: normalization constant
M_SEED = 15             #: seed for mass sample generation
POS_SEED = 38
# background values for a Gaussian distribution
BACK_MEAN = MAX_m**BETA * 1e-4      #: mean
BACK_SIGMA = BACK_MEAN * 20e-2      #: sigma
BACK_PARAM = ('Gaussian',(BACK_MEAN, BACK_SIGMA))
BACK_SEED = 1000
# detector noise values for a Gaussian distribution
NOISE_MEAN = 5e-2                   #: mean
NOISE_SIGMA = NOISE_MEAN * 50e-2    #: sigma
NOISE_PARAM = ('Gaussian',(NOISE_MEAN,NOISE_SIGMA))
NOISE_SEED = 2000
# gaussian psf
SEEING_SIGMA = 3        #: seeing
ATM_PARAM = ('Gaussian',SEEING_SIGMA)                                                                                     
##

def generate_mass_array(m_min: float = MIN_m, m_max: float = MAX_m, alpha: float = ALPHA,  sdim: int = M, seed: int = M_SEED) -> NDArray:
    """To compute masses sample from the IMF distribution

    The function takes the minimum and the maximum masses, the IMF 
    and generates a `sdim`-dimensional array of masses distributed 
    like IMF.

    The chosen method is a straightforward Monte Carlo: 
    generating uniformly random values for IMF and 
    calculating the corresponding mass.

    Parameters
    ----------
    m_min : float, optional
        the minimum mass, by default `MIN_m`
    m_max : float, optional
        the maximum mass, by default `MAX_m`
    alpha : float, optional
        the exponent of the power law, by default `ALPHA`
    sdim : int, optional
        number of stars, by default `M`
    seed : int, optional
        seed for the Monte Carlo method, by default `SEED`

    Returns
    -------
    NDArray
        array of masses distributed like IMF
    """
    # intial mass function
    IMF = lambda m : m**(-alpha)
    # evaluating IMF for the extremes
    imf_min = IMF(m_min)
    imf_max = IMF(m_max) 
    # generating the sample 
    rng = np.random.default_rng(seed=seed)
    return rng.uniform(imf_max,imf_min,sdim)**(-1/alpha)


def star_location(sdim: int = M, dim: int = N, overlap: bool = False, seed: int = POS_SEED) -> tuple[NDArray,NDArray]:
    """To locate the stars.

    It generates a list with all possible positions in the
    field matrix and draws `sdim` of those. Then it collects
    the drawn coordinates in two different arrays
    (x and y respectively).

    Parameters
    ----------
    sdim : int, optional
        number of stars, by default `M`
    dim : int, optional
        size of the field, by default `N`
    overlap : bool, optional
        if `True` overlap between stars can occur, by default `False`
    seed : int, optional
        seed of the random generator, by default `POS_SEED`

    Returns
    -------
    tuple[NDArray,NDArray]
        tuple of star coordinates arrays `X` and `Y`
    
    See Also
    --------
    numpy.random.choice : generates a random sample from a given 1-D array
    """
    # list with all possible positions in the field
    grid = [(i,j) for i in range(dim) for j in range(dim)]
    # drawing positions from grid for stars
    rng = np.random.default_rng(seed=seed)
    ind = rng.choice(len(grid), size=sdim, replace=overlap)
    # making arrays of coordinates
    X = np.array([grid[i][0] for i in ind])
    Y = np.array([grid[i][1] for i in ind])
    return (X, Y)    


def initialize(dim: int = N, sdim: int = M, masses: tuple[float, float] = (MIN_m,MAX_m), alpha: float = ALPHA, beta: float = BETA, overlap: bool = False, m_seed: int = M_SEED, p_seed: int = POS_SEED, display_fig: bool = False, **kwargs) -> tuple[NDArray, Star]:
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
    m = generate_mass_array(m_inf, m_sup, alpha=alpha, sdim=sdim, seed=m_seed)
    # evaluating corrisponding luminosities
    L = m**beta * K 
    # locating the stars
    star_pos = star_location(sdim=sdim, dim=dim, overlap=overlap, seed=p_seed)
    # updating the field matrix
    F[star_pos] += L 
    # saving stars infos
    S = Star(m,L,star_pos)
    if display_fig:
        S.plot_info(alpha,beta)       
        if 'title' not in kwargs:
            kwargs['title'] = 'Inizialized Field'
        fast_image(F,**kwargs)
    return F, S


def atm_seeing(field: NDArray, sigma: float = SEEING_SIGMA, pad_num: int = 2, bkg: DISTR | None = None, bkg_seed: int | None = BACK_SEED, display_fig: bool = False, **kwargs) -> NDArray:
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
    kernel = Gaussian(sigma).kernel()
    see_field = field_convolve(field, kernel, bkg, norm_cost=K)
    if display_fig:
        if 'title' not in kwargs:
            kwargs['title'] = 'Atmospheric Seeing '
        fast_image(see_field,**kwargs)
    # checking the field and returning it
    return see_field


def noise(distr: DISTR, dim: int = N, seed: int | None = None, display_fig: bool = False, **kwargs) -> NDArray:
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
    n = distr.field(shape=(dim,dim), seed=seed)
    if display_fig:
        fast_image(n,**kwargs)
        plt.figure()
        plt.title('Distribution')
        plt.hist(n.flatten(),len(n.flatten())//100)
        plt.show()
    if len(np.where(n < 0)) != 0:
        n = np.sqrt(n**2)
    return n * K


def add_effects(F: NDArray, background: Any, back_seed: int | None, atm_param: tuple[str, float], pad_num: int | None, det_noise: Any, det_seed: int | None, i: int, display_fig: bool = False, **kwargs):
    dim = len(F)
    # background
    kwargs['title'] = 'Background'
    n_b = noise(background,dim,seed=back_seed,display_fig=display_fig,**kwargs)
    F_b = F + n_b
    if display_fig:
        kwargs['title'] = 'Field + Background'
        fast_image(F_b,**kwargs)        
    # atm seeing
    if atm_param[0] == 'Gaussian':
        sigma = atm_param[1]
        kwargs['title'] = 'Atmospheric Seeing'
        F_bs = atm_seeing(F_b,sigma,pad_num=pad_num,bkg=background,bkg_seed=back_seed,display_fig=display_fig,**kwargs)
    # detector
    kwargs['title'] = 'Detector noise'
    n_d = noise(det_noise,dim,seed=det_seed,display_fig=display_fig,**kwargs)
    F_bsd = F_bs + n_d 
    
    if display_fig:
        kwargs = {key: kwargs[key] for key in kwargs.keys() - {'title'}}
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        fig.suptitle(f'Field Building Process : {i} acquisition')
        tmp_kwargs = {key: kwargs[key] for key in kwargs.keys() - {'v','norm'}}
        ax1.set_title('Initial Field')
        field_image(fig,ax1,F,v=1,norm='log',**tmp_kwargs)
        ax2.set_title('Background')
        field_image(fig,ax2,F_b,**kwargs)
        ax3.set_title('Seeing effect')
        field_image(fig,ax3,F_bs,**kwargs)
        ax4.set_title('Detector Noise')
        field_image(fig,ax4,F_bsd,**kwargs)
        plt.show()

        kwargs['title'] = f'Final Field: {i} acquisition'
        fast_image(F_bsd,**kwargs)        
    return F_bsd


def field_builder(acq_num: int = 3, dim: int = N, stnum: int = M, masses: tuple[float,float] = (MIN_m,MAX_m), star_param: tuple[float,float] = (ALPHA,BETA), atm_param: tuple[str,float | tuple] = ATM_PARAM, pad_num: int | None = 4, back_param: tuple[str, float | tuple] = BACK_PARAM, back_seed: int | None = BACK_SEED, det_param: tuple[str, float | tuple] = NOISE_PARAM, det_seed: int | None = NOISE_SEED, overlap: bool = False, seed: tuple[int,int] = (M_SEED, POS_SEED), iteration: int = 3, results: bool = True, display_fig: bool = False, **kwargs) -> list[Star | NDArray] | list[Star | NDArray | list[NDArray]]:
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
    m_seed, p_seed = seed
    F, S = initialize(dim,stnum,masses,*star_param,overlap=overlap,m_seed=m_seed,p_seed=p_seed,display_fig=display_fig,v=1,norm='log')
    print('\n- - - Background - - -')
    background = from_parms_to_distr(back_param,infos=True)
    print('\n- - - Detector noise - - -')
    det_noise = from_parms_to_distr(det_param, infos=True)
    print(f'\nAtm Seeing:\n{atm_param[0]} distribution')
    if atm_param[0] == 'Gaussian':
        sigma = atm_param[1]
        print(f'sigma:\t{sigma}')

    from sys import maxsize
    if isinstance(back_seed,(int,float)) or back_seed is None:
        seeds_gen = np.random.default_rng(back_seed)
        back_seed = seeds_gen.integers(maxsize,size=acq_num)
        del seeds_gen
    if isinstance(det_seed,(int,float)) or det_seed is None:
        seeds_gen = np.random.default_rng(det_seed)
        det_seed = seeds_gen.integers(maxsize,size=acq_num)
        del seeds_gen
    lights = [ add_effects(F.copy(), background, back_seed[i], atm_param, pad_num, det_noise, det_seed[i], i, display_fig=display_fig, **kwargs) for i in range(acq_num)]

    from .restoration import mean_n_std
    master_light, std_light = mean_n_std(lights, axis=0)

    if results:
        fig, (ax1,ax2) = plt.subplots(1,2)
        field_image(fig,ax1,F,v=1,norm='log')
        ax1.set_title('Starting field')
        field_image(fig,ax2,master_light)
        ax2.set_title('Master Light')

        fig, axs = plt.subplots(1,acq_num+1)
        
        for i in range(acq_num):
            axs[i].set_title(f'Light {i}')
            field_image(fig, axs[i], lights[i])
        axs[-1].set_title('Master Light')
        field_image(fig, axs[-1], master_light)
        plt.show()

    # Dark Computation
    dark_seed = np.random.default_rng(seed=det_seed[0]).integers(maxsize,size=iteration)
    dark = [noise(det_noise,dim=dim,seed=dark_seed[i]) for i in range(iteration)]
    # averaging
    master_dark, std_dark = mean_n_std(dark, axis=0) 
    if results:
        fig, ax = plt.subplots(1,iteration+1)
        for i in range(iteration):
            field_image(fig,ax[i],dark[i])
            ax[i].set_title(f'Dark {i}')
        field_image(fig,ax[-1], master_dark)
        ax[-1].set_title('Master Dark')
        plt.show()
    return S, [master_light, std_light], [master_dark, std_dark]
