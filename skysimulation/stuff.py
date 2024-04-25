from typing import Sequence
import numpy as np
from numpy.typing import NDArray

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
    def kernel(self) -> NDArray:
        """Computing a Gaussian kernel

        :param dim: size of the field
        :type dim: int
        
        :return: kernel
        :rtype: NDArray
        """
        # kernel must have an odd size
        dim = int(4*self.sigma)*2 + 1
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

def pad_field(field: NDArray, pad_size: int, bkg: DISTR, norm_cost: float = 1) -> NDArray:
    """To generate a frame of background values around the field

    Parameters
    ----------
    field : NDArray
        initial field matrix
    pad_size : int
        pixel width of the frame
    bkg : DISTR
        true/estimated background distribution
    norm_cost : float, default 1
        normalization constant for the values drwan from 
        background distribution

    Returns
    -------
    new_field : NDArray
        the field with the frame
    """
    # make a frame with negative numbers around the field
    new_field = np.pad(field, pad_size,'constant',constant_values=-1)
    # substitute negative numbers with values drawn from background distribution
    frame = new_field[new_field < 0]
    new_field[new_field < 0] = bkg.field(frame.shape)*norm_cost
    return new_field

def sqr_mask(val: float, dim: int) -> NDArray:
    return np.array([ [val, val], 
                      [val, dim - val], 
                      [dim - val, dim - val], 
                      [dim - val, val],
                      [val, val] ])

def field_convolve(field: NDArray, kernel: NDArray, bkg: DISTR, norm_cost: float = 1, mode: str = 'fft') -> NDArray:
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
    pad_size = (len(kernel)-1) // 2         #: number of pixels to pad the field
    pad_slice = slice(pad_size, -pad_size)  #: frame cut
    # pad the field to avoid edge artifacts after convolution
    tmp_field = pad_field(field, pad_size, bkg, norm_cost=norm_cost)
    # convolve
    if mode == 'fft':
        from scipy.signal import fftconvolve   
        conv_field = fftconvolve(tmp_field, kernel, mode='same')
    elif mode == '2d':
        from scipy.signal import convolve2d   
        conv_field = convolve2d(tmp_field, kernel, mode='same', boundary='fill', fillvalue=bkg.mean())
    else: raise Exception(f'Error in convolution mode!\n`{mode}` is no accepted')
    # cut and remove the frame
    return conv_field[pad_slice,pad_slice]
