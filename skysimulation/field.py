""" 
FIELD PACKAGE
=============
    This package provides all the methods to generate the field

***

::METHODS::
-----------

***

!TO DO!
-------
    - [x] ~**Implement a class `Field` to collect all the methods**~
          > It's a bad idea


***
    
?WHAT ASK TO STEVE?
-------------------
"""

import os
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from .display import fast_image, field_image
from .stuff import Gaussian, DISTR 
from .stuff import field_convolve, from_parms_to_distr, mean_n_std

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
    def __init__(self, mass: float | NDArray, lum: float | NDArray, pos: tuple[float | NDArray, float | NDArray], alpha: float, beta: float, mrange: tuple[float,float]) -> None:
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
        self.mrange = mrange
        self.alpha = alpha
        self.beta  = beta

    def lrange(self) -> tuple[float,float]:
        return (self.mrange[0]**self.beta*K,self.mrange[1]**self.beta*K)

    def mean_mass(self) -> float | None:
        if self.alpha <= 2:
            return None
        else:
            alpha = self.alpha
            minf, msup = self.mrange
            diff = lambda power: (minf**(power)-msup**(power))
            avg_mass = (alpha-1)/(alpha-2) * diff(2-alpha)/diff(1-alpha)
            return avg_mass

    def mean_lum(self) -> float | None:
        avg_mass = self.mean_mass()
        if avg_mass is None: return None
        else:
            beta = self.beta
            avg_lum = K*avg_mass**beta
            return avg_lum
        # if self.alpha <= 2:
        #     return None
        # else:
        #     alpha = self.alpha
        #     beta  = self.beta
        #     linf, lsup = self.lrange()
        #     diff = lambda power: (linf**(power)-lsup**(power))
        #     avg_mass = (1-alpha)/(1+beta-alpha) * diff((1+beta-alpha)/beta)/diff((1-alpha)/beta)
        #     return avg_mass


    def plot_info(self, sel: {'both', 'mass', 'lum'} = 'both',fontsize: int = 18) -> None:
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
        alpha = self.alpha
        beta  = self.beta
        nstars = len(self.m)    #: number of stars
        ## Mass Distribution Plot
        if 'mass' in sel or sel == 'both':
            # bins = np.linspace(min(self.m),max(self.m),nstars//3*2)
            m_cnts, m_bins = np.histogram(self.m,bins=nstars//4*3)
            masses = (m_bins[1:] + m_bins[:-1])/2
            mm = np.linspace(masses.min(),masses.max(),len(self.m))
            imf = mm**(-alpha)
            imf *= m_cnts.max()/imf.max()
            avg_mass = self.mean_mass()
            if 'mass' in sel:
                plt.figure(figsize=(12,8))
                plt.title(f'Distribution in mass of {nstars} stars\n$\\alpha = {alpha}$',fontsize=fontsize+2)
                plt.stairs(m_cnts,m_bins,fill=True)
                plt.plot(mm,imf,color='red',label='$IMF = M^{-\\alpha}$')
                if avg_mass is not None:
                    plt.axvline(avg_mass,0,1,label='$\\bar{M}='+f'{avg_mass:.2f}$')
                plt.xscale('log')
                plt.xlabel('M [$M_\\odot$]',fontsize=fontsize)
                plt.ylabel('counts',fontsize=fontsize)
                plt.legend(fontsize=fontsize)
                plt.grid()
        ## Brightness Distribution Plot
        if 'lum' in sel or sel == 'both':
            # compute the logarithm of the values
            L = np.log10(np.sort(self.lum))
            # L = np.sort(self.lum)
            l_bins = np.linspace(min(L),max(L),nstars//2)
            # bins = np.linspace(min(L),max(L),int(L.max()/L.min()))
            l_cnts, l_bins = np.histogram(L,bins=l_bins)
            brigs = (l_bins[1:] + l_bins[:-1])/2
            # log(IMF) = -a * log(M) = -a/b * log(L/K)
            ll = np.linspace(brigs.min(),brigs.max(),len(self.m))
            logimf = -alpha/beta * ll + alpha/beta * np.log10(K)
            l_imf = 10**logimf
            l_imf *= l_cnts.max()/l_imf.max()
            avg_lum = self.mean_lum()
            # imf = (bins/K)**(-alpha/beta)
            # imf *= cnts.max()/imf.max()
            if 'lum' in sel:
                plt.figure(figsize=(12,8))
                plt.title(f'Distribution in brightness of {nstars} stars\n$\\beta = {beta}$',fontsize=fontsize+2)
                plt.stairs(l_cnts,l_bins,fill=True)
                plt.plot(ll,l_imf,color='red')
                # if avg_lum is not None:
                #     plt.axvline(np.log10(avg_lum),0,1,label=f'mean={avg_lum}')
                #     plt.legend()
                plt.xlabel('$\log{(\\ell)}$',fontsize=fontsize)
                plt.ylabel('counts',fontsize=fontsize)
                plt.grid()
        if sel == 'both':
            fig, (ax1,ax2) = plt.subplots(1,2)
            fig.suptitle(f'{nstars} stars with $\\alpha = {alpha}$ and $\\beta = {beta}$',fontsize=fontsize+2)
            ax1.set_title('Distribution in mass',fontsize=fontsize+2)
            ax1.stairs(m_cnts,m_bins,fill=True)
            ax1.plot(mm,imf,color='red',label='$IMF = M^{-\\alpha}$')
            if avg_mass is not None:
                ax1.axvline(avg_mass,0,1,label='$\\bar{M}='+f'{avg_mass:.2f}$')
            ax1.set_xscale('log')
            ax1.set_xlabel('M [$M_\\odot$]',fontsize=fontsize)
            ax1.set_ylabel('counts',fontsize=fontsize)
            ax1.legend(fontsize=fontsize)
            # ax1.grid(which='both')
            ax2.set_title('Distribution in brightness',fontsize=fontsize+2)
            ax2.stairs(l_cnts,l_bins,fill=True)
            ax2.plot(ll,l_imf,color='red')
            if avg_lum is not None:
                ax2.axvline(avg_lum,0,1,label='$\\bar{\\ell}='+f'{avg_lum:.3f}$')
                # ax2.axvline(np.log10(avg_lum),0,1,label=f'mean={avg_lum}')
                ax2.legend(fontsize=fontsize)
            ax2.set_xlabel('$\log{(\\ell)}$',fontsize=fontsize)
            # ax2.grid()
            plt.figure()
            plt.plot(self.m,self.lum)
                        


### STANDARD VALUES
# MSOL = 1.989e+33 # g
# LSOL = 3.84e+33 # erg/s
N = int(1e2)            #: size of the field matrix
M = int(1e2)            #: number of stars
MIN_m = 0.5             #: min mass value for stars
MAX_m = 10              #: max mass value for stars
ALPHA = 2.35            #: IMF exp
BETA = 3                #: M-L exp
K = 1/(MAX_m**BETA)     #: normalization constant
M_SEED = 15             #: seed for mass sample generation
POS_SEED = 38
# background values for a Gaussian distribution
BACK_MEAN = MAX_m**BETA * 3.2e-4      #: mean
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
    """To inizialize the field

    It generates the stars and updates the field without any seeing 
    or noise effect.

    Parameters
    ----------
    dim : int, default N
        size of the field 
    sdim : int, default M
        number of stars
    masses : tuple[float, float], default (MIN_m,MAX_m)
        the extremes of mass range 
    alpha : float, default ALPHA
        exponent of IMF. The function is a power law 
        like Salpeter IMF -> `IMF = m**(alpha)` 
    beta : float, default BETA
        exponent of m-L relation. The function is a
        power law -> `L = m**(-beta)`
    overlap : bool, default False
        set it `True` to allow stars overlapping 
    m_seed : int, default M_SEED
        seed for the random generator of the masses
        This method uses a Monte Carlo to draw
        randomly masses from IMF distribution
    p_seed : int, default POS_SEED
        seed for the random generator to locate the 
        stars in the field uniformly        
    display_fig : bool, default False
        set it `True` to display plots 

    Returns
    -------
    F : NDArray
        inizialized field, that is an empty grid
        except for the stars
    S : Star
        class to collect all the information about
        the generated star sample 
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
    # sort values
    sortpos = np.argsort(m)[::-1]
    m = m[sortpos]
    L = L[sortpos]
    star_pos = (star_pos[0][sortpos],star_pos[1][sortpos])
    # saving stars infos
    S = Star(m, L, star_pos,alpha,beta,(m_inf,m_sup))
    if display_fig:
        S.plot_info()       
        if 'title' not in kwargs:
            kwargs['title'] = 'Inizialized Field'
        fast_image(F,**kwargs)
    return F, S


def atm_seeing(field: NDArray, sigma: float = SEEING_SIGMA, bkg: DISTR = Gaussian(BACK_SIGMA, BACK_MEAN), size: int = 4, display_fig: bool = False, **kwargs) -> NDArray:
    """To compute the atmospheric seeing

    Parameters
    ----------
    field : NDArray
        field matrix
    sigma : float, default SEEING_SIGMA
        variance root of the Gaussian distribution
    bkg : DISTR, default Gaussian(BACK_SIGMA, BACK_MEAN)
        background distribution 
    display_fig : bool, default False
        set it `True` to display plots 

    Returns
    -------
    see_field : NDArray
        field convolved with the kernel
    """
    # compute the kernel
    kernel = Gaussian(sigma).kernel(size=size)
    # convolve the field with the kernel
    see_field = field_convolve(field, kernel, bkg, norm_cost=K)
    if display_fig:
        if 'title' not in kwargs:
            kwargs['title'] = 'Atmospheric Seeing'
        fast_image(see_field,**kwargs)
    return see_field


def noise(distr: DISTR, dim: int = N, seed: int | None = None, display_fig: bool = False, **kwargs) -> NDArray:
    """To generate a noisy board

    Parameters
    ----------
    distr : DISTR
        noise distribution
    dim : int, default N
        size of the field 
    seed : int | None, default None
        seed for the random generator of the 
        noise distribution 
    display_fig : bool, default False
        set it `True` to display plots 

    Returns
    -------
    n : NDArray
        noise board
    """
    # generate the noisy board
    n = distr.field(shape=(dim,dim), seed=seed)
    if display_fig:
        fast_image(n,**kwargs)
        plt.figure()
        plt.title('Distribution')
        plt.hist(n.flatten(),len(n.flatten())//100)
        plt.show()
    # check the presence of negative values
    if len(np.where(n < 0)) != 0:
        n = np.sqrt(n**2)
    # normalize
    n *= K
    return n


def add_effects(F: NDArray, background: DISTR, back_seed: int | None, atm_param: tuple[str, float], det_noise: DISTR, det_seed: int | None, i: int = 0, add_params: dict = {}, display_fig: bool = False, **kwargs) -> NDArray:
    """To compute the final field 

    The method adds to the initialized field the effects of
    the background, the seeing and the detector noise

    Parameters
    ----------
    F : NDArray
        initialized field
    background : DISTR
        distribution of the background
    back_seed : int | None
        seed of the random generator of the
        background distribution
    atm_param : tuple[str, float]
        parameter of the atmospheric seeing
        It contains the name of the distribution
        and the value of the parameter(s)
    det_noise : DISTR
        distribution of the noise
    det_seed : int | None
        seed of the random generator of the
        noise distribution
    i : int, default 0
        acquisition number
    display_fig : bool, default False
        set it `True` to display plots 

    Returns
    -------
    F_bsd : NDArray
        final field
    """
    default_params = {
        'ker_size' : 4
        }
    for key, val in default_params.items():
        if key not in add_params.keys():
            add_params[key] = val
    dim = len(F)
    
    ## Background
    kwargs['title'] = 'Background'
    # compute the background board
    n_b = noise(background,dim,seed=back_seed,display_fig=display_fig,**kwargs)
    # add the background
    F_b = F + n_b
    if display_fig:
        kwargs['title'] = 'Field + Background'
        fast_image(F_b,**kwargs)        
    
    ## Atm Seeing
    if atm_param[0] == 'Gaussian':
        sigma = atm_param[1]
        ker_size = add_params['ker_size']
        kwargs['title'] = 'Atmospheric Seeing'
        # convolve the kernel
        F_bs = atm_seeing(F_b,sigma,bkg=background,size=ker_size,display_fig=display_fig,**kwargs)

    ## Detector
    kwargs['title'] = 'Detector noise'
    # compute the detector noise
    n_d = noise(det_noise,dim,seed=det_seed,display_fig=display_fig,**kwargs)
    # add the noise
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


def field_builder(acq_num: int = 6, dim: int = N, stnum: int = M, masses: tuple[float,float] = (MIN_m,MAX_m), star_param: tuple[float,float] = (ALPHA,BETA), atm_param: tuple[str,float | tuple] = ATM_PARAM, back_param: tuple[str, float | tuple] = BACK_PARAM, back_seed: int | None = BACK_SEED, det_param: tuple[str, float | tuple] = NOISE_PARAM, det_seed: int | None = NOISE_SEED, overlap: bool = False, seed: tuple[int,int] = (M_SEED, POS_SEED), iteration: int = 5, results: bool = True, display_fig: bool = False, **kwargs) -> tuple[Star, list[NDArray], list[NDArray]]:
    """To generate the acquired picture

    Parameters
    ----------
    acq_num : int, optional
        _description_, by default 3
    dim : int, optional
        _description_, by default N
    stnum : int, optional
        _description_, by default M
    masses : tuple[float,float], optional
        _description_, by default (MIN_m,MAX_m)
    star_param : tuple[float,float], optional
        _description_, by default (ALPHA,BETA)
    atm_param : tuple[str,float  |  tuple], optional
        _description_, by default ATM_PARAM
    back_param : tuple[str, float  |  tuple], optional
        _description_, by default BACK_PARAM
    back_seed : int | None, optional
        _description_, by default BACK_SEED
    det_param : tuple[str, float  |  tuple], optional
        _description_, by default NOISE_PARAM
    det_seed : int | None, optional
        _description_, by default NOISE_SEED
    overlap : bool, optional
        _description_, by default False
    seed : tuple[int,int], optional
        _description_, by default (M_SEED, POS_SEED)
    iteration : int, optional
        _description_, by default 3
    results : bool, optional
        _description_, by default True
    display_fig : bool, optional
        _description_, by default False

    Returns
    -------
    S : Star
        class to collect stars info
    [master_light, std_light] : [NDArray, NDArray]
        matricies of mean and STD from it of light
    [master_dark, std_dark] : [NDArray, NDArray]
        matricies of mean and STD from it of dark
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
    lights = [ add_effects(F.copy(), background, back_seed[i], atm_param, det_noise, det_seed[i], i, display_fig=display_fig, **kwargs) for i in range(acq_num)]

    master_light, std_light = mean_n_std(lights, axis=0)

    if results:
        fig, (ax1,ax2) = plt.subplots(1,2)
        field_image(fig,ax1,F,v=1,norm='log')
        ax1.set_title('Source Image',fontsize=20)
        field_image(fig,ax2,master_light)
        ax2.set_title('Master Light',fontsize=20)

        cols = acq_num//2 if acq_num%2==0 else acq_num//2+1
        fig, axs = plt.subplots(2,cols)
        vmin = np.min([ lights[i].min() for i in range(acq_num)])
        vmax = np.max([ lights[i].max() for i in range(acq_num)])
        colorbar = {'colorbar': False, 'colorbar_pos': 'bottom'}
        for i in range(acq_num):
            index = (i//cols,i%cols)
            # if i%2 == 1: colorbar['colorbar'] = True  
            axs[index].set_title(f'Light Frame {i}',fontsize=20)
            if i == acq_num-1: colorbar['colorbar'] = True
            field_image(fig, axs[index],lights[i],vmin=vmin,vmax=vmax,**colorbar)
            # colorbar['colorbar'] = False
        plt.show()

    # Dark Computation
    dark_seed = np.random.default_rng(seed=det_seed[0]).integers(maxsize,size=iteration)
    dark = [noise(det_noise,dim=dim,seed=dark_seed[i]) for i in range(iteration)]
    # averaging
    master_dark, std_dark = mean_n_std(dark, axis=0) 
    if results:
        vmin = np.min([ dark[i].min() for i in range(iteration)]+[master_dark.min()])
        vmax = np.max([ dark[i].max() for i in range(iteration)]+[master_dark.max()])
        fig, ax = plt.subplots(2,iteration//2+1)
        cols = iteration//2 if iteration%2==0 else iteration//2+1
        for i in range(iteration):
            index = (i//cols,i%cols)
            field_image(fig,ax[index],dark[i], vmin=vmin,vmax=vmax,colorbar=False)
            ax[index].set_title(f'Dark {i}',fontsize=20)
        colorbar = {'colorbar': True, 'colorbar_pos': 'bottom'}
        field_image(fig,ax[1,-1], master_dark, vmin=vmin,vmax=vmax,**colorbar)
        ax[1,-1].set_title('Master Dark',fontsize=20)
        plt.show()
    return S, [master_light, std_light], [master_dark, std_dark]
