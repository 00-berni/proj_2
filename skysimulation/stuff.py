import os
from typing import Sequence, Any, Literal
import numpy as np
from numpy.typing import NDArray,ArrayLike
from astropy.units import Quantity
import matplotlib.pyplot as plt
import pandas as pd

### CONSTANTS
## Paths
PWD = os.path.dirname(os.path.realpath(__file__))           #: path of the current dir
PROJECT_DIR = os.path.split(PWD)[0]                         #: path of the project dir
RESULT_DIR = os.path.join(PROJECT_DIR, 'result_data')       #: path of results dir


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
    def kernel(self,size: int = 4) -> NDArray:
        """Computing a Gaussian kernel

        :param dim: size of the field
        :type dim: int
        
        :return: kernel
        :rtype: NDArray
        """
        # kernel must have an odd size
        dim = int(size*self.sigma)*2 + 1
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
        
DISTR = Gaussian | Uniform | Poisson    #: variable to collect distributions type

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

def distance(p1: tuple[ArrayLike,ArrayLike], p2: tuple[ArrayLike,ArrayLike]) -> ArrayLike:
    x1, y1 = p1 
    x2, y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def pad_field(field: NDArray, pad_size: int, bkg: DISTR | float, norm_cost: float = 1) -> NDArray:
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
    if isinstance(bkg,(float,int)):
        new_field[new_field < 0] = bkg*norm_cost
    else:
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
    pad_size = (len(kernel)-1) // 2   +2      #: number of pixels to pad the field
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

def mean_n_std(data: Sequence[Any], axis: int | None = None, weights: Sequence[Any] | None = None) -> tuple[float, float]:
    """To compute the mean and standard deviation from it

    Parameters
    ----------
    data : Sequence[Any]
        values of the sample
    axis : int | None, default None
        axis over which averaging
    weights : Sequence[Any] | None, default None
        array of weights associated with data

    Returns
    -------
    mean : float
        the mean of the data
    std : float
        the STD from the mean
    """
    dim = len(data)     #: size of the sample
    # compute the mean
    mean = np.average(data,axis=axis,weights=weights)
    # compute the STD from it
    if weights is None:
        std = np.std(data, axis=axis, ddof=1)
        # std = np.sqrt( ((data-mean)**2).sum(axis=axis) / (dim*(dim-1)) )
    else:
        std = np.sqrt(np.average((data-mean)**2, weights=weights) / (dim-1) * dim)
    return mean, std

def autocorr(arr: NDArray, **kwargs) -> float | NDArray:
    from scipy.signal import correlate
    return correlate(arr,arr,**kwargs)

def peak_pos(field: NDArray) -> int | tuple[int,int]:
    """Finding the coordinate/s of the maximum

    Parameters
    ----------
    field : NDArray
        the frame

    Returns
    -------
    int | tuple[int,int]
        position index(es) of the maximum
    """
    if len(field.shape) == 1:
        return field.argmax()
    else:
        return np.unravel_index(field.argmax(),field.shape)

def minimum_pos(field: NDArray) -> int | tuple[int,int]:
    """Finding the coordinate/s of the minimum

    Parameters
    ----------
    field : NDArray
        the frame

    Returns
    -------
    int | tuple[int,int]
        position index(es) of the minimum
    """
    if len(field.shape) == 1:
        return field.argmin()
    else:
        return np.unravel_index(field.argmin(),field.shape)

def magnitude_order(number: ArrayLike) -> ArrayLike:
    number = np.abs(number) 
    order = np.floor(np.log10(number)).astype(int)
    return order

def unc_format(value: ArrayLike, err: ArrayLike) -> list[str]:
    err_ord = magnitude_order(err).min()
    val_ord = magnitude_order(value).max()
    order = val_ord - err_ord 
    fmt = [f'%.{order:d}e',r'%.0e']
    return fmt
    
def print_measure(value: float | Quantity, err: float | Quantity, name: str = 'value', unit: str = '') -> None:
    if isinstance(value, Quantity) and isinstance(err, Quantity):
        unit = value.unit.to_string()
        value = value.value
        err = err.value
    fmt = unc_format(value,err)
    if value != 0:
        fmt = name + ' = {value:' + fmt[0][1:] + '} +/- {err:' + fmt[1][1:] + '} ' + unit + ' ---> {perc:.2%}'
        print(fmt.format(value=value,err=err,perc=err/value))
    else:
        fmt = name + ' = {value:' + fmt[0][1:] + '} +/- {err:' + fmt[1][1:] + '} ' + unit 
        print(fmt.format(value=value,err=err))

def dist_corr(postions: tuple[NDArray,NDArray], binning: int = 63,fontsize: int = 18, display_plots: bool = False) -> NDArray:
    xpos, ypos = postions
    distances = [np.sqrt((xpos[i+1:]-xpos[i])**2 + (ypos[i+1:]-ypos[i])**2) for i in range(len(xpos)-1)]
    distances = np.concatenate(distances)
    if display_plots:
        print('Mean Distance',np.mean(distances),'px')
        plt.figure()
        plt.title('Distances distribution',fontsize=fontsize+2)
        plt.hist(distances,binning,histtype='step',density=True)
        plt.axvline(np.mean(distances),0,1,color='orange',linestyle='dashed',label='mean')
        plt.legend(fontsize=fontsize)
        plt.show()
    return distances

def store_results(file_name: str, data: ArrayLike, main_dir: str | None = None, columns: ArrayLike | None = None, **csvkw) -> None:
    """To store results in a `.txt` file

    Parameters
    ----------
    file_name : str
        name of the file
    data : ArrayLike
        data to store
    ch_obs : str
        chosen observation
    ch_obj : str
        chosen target name
    **txtkw
        parameters of `numpy.savetxt()`
        the parameter `'delimiter'` is set to `'\t'` by default
    """
    # check delimiter
    if 'sep' not in csvkw.keys():
        csvkw['sep'] = ','
    # build the path
    res_dir = RESULT_DIR
    if main_dir is not None:
        new_dir = os.path.join(res_dir,main_dir)
        if not os.path.isdir(new_dir): os.mkdir(new_dir)
        res_dir = new_dir
    file_path = os.path.join(res_dir, file_name + '.csv')
    # save data
    dataframe = pd.DataFrame(np.column_stack(data),columns=columns,copy=True)
    dataframe.to_csv(file_path,**csvkw)

def open_data(file_name: str, main_dir: str = '', out_type: Literal['dataframe','array'] = 'dataframe',**csvtw) -> pd.DataFrame | NDArray:
    file_path = os.path.join(RESULT_DIR,main_dir,file_name+'.csv')
    data = pd.read_csv(file_path,**csvtw)
    if out_type == 'dataframe':
        return data
    elif out_type == 'array':
        return data.to_numpy().transpose()[1:]
    
def log_update(log_txt: str, file_name: str = '', main_dir: str = '', mode: str = 'a', sep: str = '\n',**openkws) -> None:
    # build the path
    res_dir = os.path.join(RESULT_DIR,main_dir)
    if not os.path.isdir(res_dir): os.mkdir(res_dir)
    file_name = 'log' if file_name == '' else 'log-' + file_name
    file_path = os.path.join(res_dir, file_name + '.txt')
    
    log_file = open(file_path,mode=mode,**openkws)
    log_file.write(log_txt+sep)
    log_file.close()
