""" 
RESTORATION PACKAGE
===================
    This package provides all the methods to recover the field

***

::METHODS::
-----------
    - [x] **Background Estimation**
    - [x] **Object Extraction**
    - [x] **Kernel Fit**
    - [x] **RL Algorithm**
    - [] **Object Detector** 
    - [] **Light Recovery**

***

!TO DO!
-------
    - [] Try to understand the meaning of weighted std and **find some references**
    - [] Documentation
    - [] Check the meaning of all actions
    - [x] Add the exact uncertainties to the objects
    - [] Look for a use of the [-1] size check 
    - [] **Investigate better the different results for different seeds**
    - [] **Understand the squared artifact due to RL**

***
    
?WHAT ASK TO STEVE?
-------------------
    - [] ***What's the role of binning? Is mine correct or not?***
    - [] *Is it good to use the brightness as a weight?*
    - [] *Is it ok to use this method for the fit?*
    - [] ***Is the RL output good?***
"""


from typing import Callable, Sequence, Any, TypeVar, Literal
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray,ArrayLike
from scipy.signal import find_peaks
from .display import fast_image, field_image
from .field import Gaussian, field_convolve, N, Uniform, noise, NOISE_SEED, DISTR



class FuncFit():
    """To compute the fit procedure of some data

    Attributes
    ----------
    data : list[NDArray | None]
        the x and y data and (if there are) their uncertanties
    fit_par : NDArray | None
        fit estimated parameters
    fit_err : NDArray | None
        uncertanties of `fit_par`
    res : dict
        it collects all the results  
    """
    def __init__(self, xdata: Any, ydata: Any, yerr: Any = None, xerr: Any = None) -> None:
        """Constructor of the class

        Parameters
        ----------
        xdata : Any
            x data points
        ydata : Any
            y data points
        yerr : Any, default None
            if there is, the uncertainties of `ydata` 
        xerr : Any, default None
            if there is, the uncertainties of `xdata` 
        """
        self.data = [xdata, ydata, yerr, xerr]
        self.fit_par: NDArray | None = None
        self.fit_err: NDArray | None = None
        self.res = {}


    def fit(self, method: Callable[[Any,Any],Any], initial_values: Sequence[Any],**kwargs) -> None:
        """To compute the fit

        Parameters
        ----------
        method : Callable[[Any,Any],Any]
            the fit function
        initial_values : Sequence[Any]
            initial values
        """
        # importing the function
        xdata, ydata = self.data[:2]
        sigma = self.data[2]
        Dx = self.data[3]
        self.res['func'] = method
        from scipy import odr
        def fit_model(pars, x):
            return method(x, *pars)
        model = odr.Model(fit_model)
        data = odr.RealData(xdata,ydata,sx=Dx,sy=sigma)
        alg = odr.ODR(data, model, beta0=initial_values)
        out = alg.run()
        pop = out.beta
        pcov = out.cov_beta

        # computing the fit
        # from scipy.optimize import curve_fit
        # pop, pcov = curve_fit(method, xdata, ydata, initial_values, sigma=sigma,**kwargs)
        # extracting the errors
        Dpop = np.sqrt(pcov.diagonal())
        self.fit_par = pop
        self.fit_err = Dpop
        self.res['cov'] = pcov
        # if sigma is not None:
        #     # evaluating function in `xdata`
        #     fit = method(xdata,*pop)
        #     # computing chi squared
        #     chisq = (((ydata-fit)/sigma)**2).sum()
        #     chi0 = len(ydata) - len(pop)
        #     self.res['chisq'] = (chisq, chi0)
        chisq = out.sum_square
        chi0 = len(ydata) - len(pop)
        self.res['chisq'] = (chisq, chi0)
    
    def infos(self, names: list[str] | None = None) -> None:
        """To plot information about the fit

        Parameters
        ----------
        names : list[str] | None, default None
            list of fit parameters names
        """
        pop  = self.fit_par
        Dpop = self.fit_err
        print('\nFit results:')
        if names is None:
            names = [f'par{i}' for i in range(len(pop))]
        for name, par, Dpar in zip(names,pop,Dpop):
            print(f'\t{name}: {par:.2} +- {Dpar:.2}')
        chisq, chi0 = self.res['chisq']
        print(f'\tred_chi = {chisq/chi0*100:.2f} +- {np.sqrt(2/chi0)*100:.2f} %')

    def results(self) -> tuple[NDArray, NDArray] | tuple[None, None]:
        return self.fit_par, self.fit_err

    def pipeline(self,method: Callable[[Any,Any],Any], initial_values: Sequence[Any], names: list[str] | None = None,**kwargs) -> None:
        self.fit(method=method,initial_values=initial_values,**kwargs)
        self.infos(names=names)
    
    def gaussian_fit(self, intial_values: Sequence[Any], names: list[str] | None = None,**kwargs) -> None:
   
        def gauss_func(data: float | NDArray, *args) -> float | NDArray:
            k, mu, sigma = args
            z = (data - mu) / sigma
            return k * np.exp(-z**2/2)

        if names is None: 
            names = ['k','mu','sigma']
        self.pipeline(method=gauss_func,initial_values=intial_values,names=names,**kwargs)
    
class Histogram():

    def __init__(self, data: NDArray, bins: int | NDArray) -> None:
        
        self.data = data.copy()
        self.bins = bins
        self.cnts = None
        self.yerr = None
    
    def hist(self, **kwargs) -> tuple[NDArray, NDArray]:    
        return np.histogram(self.data, bins=self.bins, **kwargs)

    def errhist(self, err: NDArray, **kwargs) -> tuple[NDArray, NDArray, tuple[NDArray, NDArray]]:
        def in_bin(data: NDArray, edges: NDArray, idx: int) -> NDArray:
            cond1 = data >= edges[idx]
            if idx == len(edges)-2:
                cond2 = data <= edges[idx+1]
            else:
                cond2 = data < edges[idx+1]
            return cond1 & cond2

        rx_data = self.data + err
        lx_data = self.data - err
        bins = self.bins
        if isinstance(bins, int):
            max_val = rx_data.max()          
            min_val = lx_data.min()
            bins = np.linspace(min_val, max_val, bins+1, endpoint=True)
    

        cnts, _ = np.histogram(self.data,bins=bins,**kwargs)
        up_cnts = np.array([len(np.where( in_bin(rx_data,bins,i) | in_bin(lx_data,bins,i)  )[0]) for i in range(len(bins)-1)])            
        dw_cnts = np.array([len(np.where( in_bin(rx_data,bins,i) & in_bin(lx_data,bins,i)  )[0]) for i in range(len(bins)-1)])
        return cnts, bins, (up_cnts, dw_cnts)            

    def histogram(self, err: NDArray | None = None, **kwargs) -> None:
        if err is None:
            self.cnts, self.bins = self.hist(**kwargs)
        else:
            self.cnts, self.bins, self.yerr = self.errhist(err,**kwargs)

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
        std = np.sqrt( ((data-mean)**2).sum(axis=axis) / (dim*(dim-1)) )
    else:
        std = np.sqrt(np.average((data-mean)**2, weights=weights) / (dim-1) * dim)
    return mean, std


def bkg_est(field: NDArray, binning: int | Sequence[int | float] | None = None, display_plot: bool = False) -> tuple[float,float]:
    """To estimate the background brightness

    Assuming the distribution of the background is a normal one,
    the method estimates the mean and the variance root 
    respectively from the highest count value and the HWHM of the 
    histogram of the data 
    
    If there are more peaks than one, the methods averages between
    them 

    Parameters
    ----------
    field : NDArray
        field matrix
    binning : int | Sequence[int  |  float] | None, optional
        this variable is related with the bins of the histogram: 
          - if `int` it is the number of bins 
          - if `Sequence` type it is sequence of the edges of 
            the bins 
          - if `None`, as by default, the binning is computed 
            from the magnitude between the minimum and maximum 
            values 
    display_plot : bool, optional
        parameter to plot data, by default `False`

    Returns
    -------
    mean_bkg : float
        the mean of the estimated normal distribution
    sigma_bkg : float
        the variance root of the estimated normal distribution
    
    Notes
    -----
    In case of multiple peaks, the method estimates first the 
    parameters obtained from the highest, then checks the 
    presence of other maxima only over the height at which the 
    distribution is at a :math:`\sigma` from the mean

    The method calls the function `find_peaks()` of the package 
    `scipy.signal` to look for peaks 
    """
    ## Initialization
    frame = field.copy()        #: copy of the field matrix
    data = frame.flatten()      #: 1-D data array
    if binning is None:
        # compute the magnitude between max and min data
        ratio = int(data.max()/data.min())
        # set the number of bins
        binning = ratio*10
        # print information
        print('\nBinning Results')
        print('Number of pixels:', data.shape[0])
        print('Magnitudes:', ratio)
        print('Number of bins:', binning)
    
    ## Parameters Estimation
    # compute the histogram
    cnts, bins = np.histogram(data,bins=binning)
    # find the index of the max value
    max_indx = cnts.argmax()
    # compute the corresponding brightness value
    mean_bkg = (bins[max_indx+1] + bins[max_indx])/2
    # compute the half maximum value    
    hm = cnts[max_indx]/2
    # find the value that approximates better `hm`
    hm_indx = abs(cnts - hm).argmin()
    # compute HWHM
    hwhm = abs((bins[hm_indx+1] + bins[hm_indx])/2 - mean_bkg)
    # compute variance root assuming a normal distribution
    sigma_bkg = hwhm/np.sqrt(2*np.log(2))
    
    ## Peaks Check
    # compute the height at which the distribution is at a sigma from the mean   
    sigma_indx = abs(bins - (mean_bkg-sigma_bkg)).argmin()
    sigma_height = cnts[sigma_indx]*2
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(cnts,height=sigma_height)
    # compute again the parameters in case of multiple peaks
    if len(peaks) > 1:
        # average between peaks
        mean_bkg = np.array([(bins[pk+1]+bins[pk])/2 for pk in peaks]).mean()
        hm = cnts[peaks].mean()/2
        hm_indx = abs(cnts - hm).argmin()
        hwhm = abs((bins[hm_indx+1] + bins[hm_indx])/2 - mean_bkg)
        sigma_bkg = hwhm/np.sqrt(2*np.log(2))
    # print the results
    print('\nBackground estimation')
    print(f'\tmean:\t{mean_bkg}\n\tsigma:\t{sigma_bkg}\n\trelerr:\t{sigma_bkg/mean_bkg*100:.2f} %')
    
    ## Plotting
    if display_plot:
        # define variables to plot the estimated distribution
        xx = np.linspace((bins[1]+bins[0])/2,(bins[-1]+bins[-2])/2,len(data))
        model = Gaussian(sigma_bkg, mean_bkg).value(xx-mean_bkg) *cnts.max()
        plt.figure()
        plt.stairs(cnts, bins, fill=False, label='data')
        plt.axvline(mean_bkg, 0, 1, color='red', linestyle='dotted', label='estimated mean')
        plt.plot(xx,model, label='estimated gaussian')
        plt.legend()
        plt.show()

    return mean_bkg, sigma_bkg

def new_moving(direction: str, field: NDArray, index: tuple[int,int], back: float, size: int = 7, debug_check: bool = False) -> list[int] | int:
    """To compute the size in one direction

    Parameters
    ----------
    direction : str
        the chosen direction
    field : NDArray
        the field matrix
    index : tuple[int,int]
        the coordinates of the brightest pixel
    back : float
        the estimation of mean background value
    size : int, optional
        maximum size of the object, by default 7
    debug_check : bool, optional
        . . ., by default False

    Returns
    -------
    results : list[int] | int
        size(s) in the chosen direction
    """
    dim = len(field)        #: size of the field
    x,y = index             #: coordinates of the brightest pixel
    results = []            #: list to collect results
    hm = field[index]/2     #: half maximum
    
    ## Initialization
    #> inizialize the variable for the direction
    #>    1 : forward
    #>    0 : ignored
    #>   -1 : backward
    xd = 0
    yd = 0
    # initialize the limits
    xmax = x
    ymax = y
    # initialize the condition on x and y to stop
    xcond = lambda *args: True
    ycond = lambda *args: True
    
    ## Along x Direction
    if 'fx' in direction:   #: forward
        #?
        if debug_check:
            print('hey')
        #?
        # compute the edge
        xmax = min(size, dim-1-x)
        if xmax == 0:       #: impossible movement case
            results += [0]
        else:               #: update the condition on x instead
            xd = 1
            xcond = lambda xval, xlim: xval < xlim
    elif 'bx' in direction: #: backward 
        # compute the edge
        xmax = min(size, x)
        if xmax == 0:       #: impossible movement case
            results += [0]
        else:               #: update the condition on x instead
            xd = -1
            xcond = lambda xval, xlim: xval < xlim 
    
    ## Along y Direction
    if 'fy' in direction:   #: forward
        # compute the edge
        ymax = min(size, dim-1-y)        
        if ymax == 0:       #: impossible movement 
            results += [0]
        else:               #: update the condition on y
            yd = 1
            ycond = lambda yval, ylim: yval < ylim  
    elif 'by' in direction: #: backward
        # compute the edge
        ymax = min(size, y)
        if ymax == 0:       #: impossible movement 
            results += [0]
        else:               #: update the condition on y
            yd = -1
            ycond = lambda yval, ylim: yval < ylim 
    
    ## Steps Routine
    if xd != 0 or yd != 0:  #: if there are no forbidden directions
        # inizilize the variables for the size
        xsize = 1
        ysize = 1
        #?
        if debug_check:
            print(xcond(xsize,xmax),xmax)
        #?
        while (xcond(xsize,xmax) and ycond(ysize,ymax)):
            # compute the step
            step = field[x + xd*xsize, y + yd*ysize]
            if step == 0 or step < back:
                # store the value
                if 'x' in direction and xd != 0: results  = [xsize] + results
                if 'y' in direction and yd != 0: results += [ysize]
                if len(results) == 1: results = results[0]
                return results
            elif step <= hm:
                # compute the gradient with the previuos pixel
                step0 = field[x + xd*(xsize-1), y + yd*(ysize-1)]
                grad = step0 - step
                if grad < 0:
                    # store the value
                    if 'x' in direction and xd != 0: results  = [xsize-1] + results
                    if 'y' in direction and yd != 0: results += [ysize-1]
                    if len(results) == 1: results = results[0]
                    return results
            # go on to the next step
            xsize += 1
            ysize += 1
        #! CHECK 
        if 'x' in direction and xd != 0: results  = [-1] + results
        if 'y' in direction and yd != 0: results += [-1]
    # check the results
    if len(results) == 0: raise
    if len(results) == 1: results = results[0]
    return results

def new_grad_check(field: NDArray, index: tuple[int,int], back: float, size: int = 7, debug_check: bool = False) -> NDArray:
    """To compute the size of an object

    Parameters
    ----------
    field : NDArray
        the field matrix
    index : tuple[int,int]
        the coordinates of the brightest pixel
    back : float
        the estimation of mean background value
    size : int, optional
        maximum size of the object, by default 7
    debug_check : bool, optional
        . . ., by default False

    Returns
    -------
    a_xysize : NDArray
        object sizes. 
        It is a matrix 2x2:
            [ [xsize_inf, xsize_sup]
              [ysize_inf, ysize_sup] ]
    """
    # define a method to move out from the brightest pixel
    n_mov = lambda val : new_moving(val,field,index,back,size)
    xf_dir = ['fx','bx','fy','by']      #: list of all directions
    # compute the size in each directions of `xf_dir`
    a_xysize = np.array([n_mov(dir) for dir in xf_dir]) 
    #?
    if debug_check:
        print('A_XF',type(a_xysize))
    #?
    # check the 
    n_pos = np.where(a_xysize == -1)[0]
    if len(n_pos) != 0:
        dim = len(field)    #: size of the field
        # compute the size in each direction 
        a_xysize[n_pos] = np.array([ min(size, ((i+1)%2)*dim + (i%2*2-1)*(index[i//2])) for i in n_pos])
        #?
        if debug_check:
            print('A_XF',type(a_xysize),a_xysize)
        #?
    # a_xysize = np.where(a_xysize > size, a_xysize-size, a_xysize)
    return a_xysize.reshape(2,2)

def corr_check(obj: NDArray, numpeak: int = 3, display_plot: bool = False,**kwargs) -> bool:
    from scipy.signal import find_peaks
    rows = obj.flatten()
    r_corr = autocorr(rows,**kwargs)
    r_peak, _ = find_peaks(r_corr)
    cols = np.stack(obj,axis=-1).flatten()
    c_corr = autocorr(cols,**kwargs)
    c_peak, _ = find_peaks(c_corr)
    if len(r_peak) == 1 and len(c_peak) == 1 and display_plot:
        app = np.append(rows,cols)
        app_corr = autocorr(app,**kwargs)
        app_peak, _ = find_peaks(app_corr)
        s_corr = autocorr((rows*cols)**3,**kwargs)
        s_peak, _ = find_peaks(s_corr)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(app_corr,'.-')
        plt.plot(app_peak,app_corr[app_peak],'xr')
        plt.subplot(1,2,2)
        plt.plot(s_corr,'.-')
        plt.plot(s_peak,s_corr[s_peak],'xr')
    if display_plot:
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(rows,'.-')
        plt.subplot(2,2,2)
        plt.plot(r_corr,'.-')
        if len(r_peak) > 0:
            plt.plot(r_peak,r_corr[r_peak],'x',color='red')
        plt.subplot(2,2,3)
        plt.plot(cols,'.-')
        plt.subplot(2,2,4)
        plt.plot(c_corr,'.-')
        if len(c_peak) > 0:
            plt.plot(c_peak,c_corr[c_peak],'x',color='red')

        fast_image(obj)
    if max(len(r_peak),len(c_peak)) < numpeak:
        return False
    else:
        return True



def selection(obj: NDArray, index: tuple[int,int], apos: NDArray, size: int, sel: Literal["all", "size", "dist"] = 'all', mindist: int = 5, minsize: int = 3, debug_check: bool = False) -> bool:
    """To check whether a selected object satisfies some required conditions

    Parameters
    ----------
    obj : NDArray
        selected object to investigate
    index : tuple[int,int]
        coordinates of the brightest pixel of the object
    apos : NDArray
        array of the coordinates of the objects extracted before `obj`
    size : int
        the maximum size of an object
    sel : {'all', 'size', 'dist'}, default 'all'
        parameter to select the kind of check to compute.
        Only 3 options are accepted:
          * `'size'`:   method checks the object size do not exceed 
                        the chosen maximum. It requires the 
                        `'minsize'` parameter
          * `'dist'`:   method checks the distance of the object from 
                        each previous extracted object is higher than the chosen
                        minimum. It requires the `'mindist'` parameter
          * `'all'` :   method computes both checks
    mindist : int, default 5
        minimum required distance between objects
    minsize : int, default 3
        maximum required size of an object
    debug_check : bool, optional
        _description_, by default False

    Returns
    -------
    bool
        condition to accept (`True`) or reject (`False`) the selected
        object

    Raises
    ------
    Exception
        _description_
    Exception
        _description_
    """
    cond = False                #: variable to check if `sel` is 'all', 'size' or 'dist'
    xdim, ydim = obj.shape      #: sizes of the object 
    if sel == 'all' or 'size' in sel:   #: size check
        # update the warning variable
        cond = True
        # take the minimum size of the object
        dim = min(xdim,ydim)
        if size != 1 and dim <= minsize: 
            if debug_check:
                print(f'\t:Selection:\tdim = {dim}')
            return False
    if sel == 'all' or 'dist' in sel:   #: distance check 
        # update the warning variable
        cond = True
        # check the distance only if other objects are passed
        if len(apos[0]) > 0:    
            # check distance is not negative 
            if (size+mindist) < 0: raise Exception(f'ErrorValues: the `size + mindist` is negative, change the values')
            # define method to compute the length of a segment
            dist = lambda x,y: np.sqrt(x**2 + y**2)
            x, y = index                    #: coordinates of the brightest pixel
            xi, yi = np.copy(apos[:,:])     #: coordinates of extracted objects
            # update the minimum distance adding the maximum size of an object
            mindist += size
            # compute the distance from each extracted object
            adist = np.array( [dist(xi[i]-x, yi[i]-y) for i in range(len(xi))] )
            # check the presence of too near object
            pos = np.where(np.logical_and(adist <= mindist, adist != 0))  
            if len(pos[0]) != 0:
                    if debug_check:
                        print(f'\t:Selection:\tdist = {adist[pos]} - adist = {adist} - mindist = {mindist}')
                    return False
            # delete useless variables
            del x, y, xi, yi, dist, adist, pos
    # check the value assigned to `sel`
    if not cond: raise Exception(f'ErrorValue: `sel`={sel} is not allowed')
    return True    


##*
def object_isolation(field: NDArray, thr: float, sigma: NDArray, size: int = 5, objnum: int = 10, reshape: bool = False, reshape_corr: bool = False, sel_cond: bool = False, mindist: int = 5, minsize: int = 3, cutsize: int = 5, numpeaks: int = 2, grad_new: bool = True, corr_cond: bool = True, debug_check: bool = False, results: bool = True, display_fig: bool = False,**kwargs) -> tuple[list[NDArray], list[NDArray], NDArray] | None:
    """To isolate the most luminous star object.
   
    The function calls the `size_est()` function to compute the size of the object and
    then to extract it from the field.


    Parameters
    ----------
    field : NDArray
        the field matrix
    thr : float
        value below which searching routine stops, 
        e.g. the average value of the background
    sigma : NDArray
        STD of field pixels
    size : int, default 5
        maximum size of an extracted object
    objnum : int, default 10
        maximum number of object to extract
    reshape : bool, optional
        _description_, by default False
    reshape_corr : bool, optional
        _description_, by default False
    sel_cond : bool, default False
        _description_
    mindist : int, default 5
        _description_ 
    minsize : int, default 3
        _description_
    cutsize : int, optional
        _description_, by default 5
    numpeaks : int, optional
        _description_, by default 2
    grad_new : bool, optional
        _description_, by default True
    corr_cond : bool, optional
        _description_, by default True
    debug_check : bool, optional
        _description_, by default False
    display_fig : bool, default False
        parameter to show figures

    Returns
    -------
    obj : list[NDArray]
        list of extracted objects    
    err : list[NDArray]
        list of STD matrix for each 
        extracted object
    pos : NDArray
        array of the coordinates of
        all objects
        The format is 
            `[ [x_array], [y_array]  ]`
    """
    # make a copy of the field
    tmp_field = field.copy()        #: field from which objects will be removed
    display_field = field.copy()    #: field from which only selected objects will be removed (for plotting only)
    # initialize variables to collect data
    a_pos = np.empty(shape=(2,0),dtype=int)         #: array to store coordinates of all objects
    sel_obj = [[],[]]                               #: list to collect accepted objects 
    sel_pos = np.empty(shape=(2,0),dtype=int)       #: array to store coordinates of `sel_obj`
    rej_obj = [[],[]]                               #: list to collect rejected objects
    rej_pos = np.empty(shape=(2,0),dtype=int)       #: array to store coordinates of `rej_obj`
    if display_fig:
        tmp_kwargs = {key: kwargs[key] for key in kwargs.keys() - {'title'}} 
    
    ## Extraction Routine
    k = 0           #: counter of selected objects
    ctrl_cnt = 0    #: counter of iterations
    print('\n- - - Object Extraction - - -')
    while k < objnum:
        # find the position of the brightest pixel
        index = peak_pos(tmp_field)
        
        #? per me
        if debug_check:
            ctrl = False    #?: solo per me
            if 0 in index: 
                print(index)
                ctrl = True
        #?
        
        # store the brightest pixel value
        peak = tmp_field[index]
        if peak <= thr:     #: stopping condition
            break
        # compute the size
        a_size = new_grad_check(tmp_field,index,thr,size)
        
        #? per me
        if debug_check:
            if ctrl: 
                print(index)
                print(a_size)
        #?
                
        print(f':: Iteration {k} of object_isolation :: ')
        print('a_size',a_size)
        x, y = index                        #: coordinates of the brightest pixel
        xu, xd, yu, yd = a_size.flatten()   #: edges of the object
        # compute slices for edges
        xr = slice(x-xd, x+xu+1) 
        yr = slice(y-yd, y+yu+1)
        # remove the object from the field
        tmp_field[xr,yr] = 0.0
        if debug_check:
            print('Slices: ',xr,yr)
            print('a_size 2',a_size)
        # update the array for coordinates
        a_pos = np.append(a_pos,[[x],[y]],axis=1)
        if any(([[0,0],[0,0]] == a_size).all(axis=1)):  #: a single pixel is not accepted
            # storing the object and its coordinates
            rej_obj += [field[xr,yr].copy()]
            rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
            if debug_check:
                print(f'Rejected obj: ({x},{y})')
        else:            
            # store the object and its std
            obj = field[xr,yr].copy() 
            err = sigma[xr,yr].copy()
            # check the condition to accept an object
            save_cond = selection(obj,index,a_pos,size,sel='all',mindist=mindist,minsize=minsize) if sel_cond else True
            if save_cond:       #: accepted object
                if debug_check:
                    print(f'** OBJECT SELECTED ->\t{k}')
                # remove it from the field for displaying
                display_field[xr,yr] = 0.0
                # store the selected object, its std and coordinates
                sel_obj[0] += [obj]
                sel_obj[1] += [err]
                sel_pos = np.append(sel_pos,[[x],[y]],axis=1)
                if display_fig:
                    tmp_kwargs['title'] = f'N. {k+1} object {index} - {ctrl_cnt}'
                # update the counter of object
                k += 1 
            else: 
                if debug_check:
                    print(f'!! OBJECT REJECTED ->\t{k}')
                # store the rejected object, its std and coordinates
                rej_obj[0] += [obj]
                rej_obj[1] += [err]
                rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
                if display_fig:
                    tmp_kwargs['title'] = f'Rejected object {index} - {ctrl_cnt}'
            if display_fig:     #: show the object
                fig, ax = plt.subplots(1,1)
                field_image(fig,ax,tmp_field,**{key: tmp_kwargs[key] for key in tmp_kwargs.keys() - {'title'}})
                if len(rej_pos[0])!= 0:
                    ax.plot(rej_pos[1,:-1],rej_pos[0,:-1],'.r')
                    ax.plot(rej_pos[1,-1],rej_pos[0,-1],'x',color='red')
                if len(sel_pos[0])!=0:
                    ax.plot(sel_pos[1,:-1],sel_pos[0,:-1],'.b')
                    ax.plot(sel_pos[1,-1],sel_pos[0,-1],'xb')
                fast_image(obj,**tmp_kwargs) 
            # check the condition to prevent the stack-overflow
            if (ctrl_cnt >= objnum and len(sel_obj[0]) >= 3) or ctrl_cnt > 2*objnum and sel_cond:
                break
            # update the counter of iterations
            ctrl_cnt += 1

    # display the field after extraction
    if results:
        if 'title' not in kwargs:
            kwargs['title'] = 'Field after extraction'
        fast_image(display_field,**kwargs)
        fast_image(tmp_field,**kwargs)    
    # show the field with markers for rejected and selected objects
    if sel_cond and len(sel_obj[0]) > 0 and results:
        fig, ax = plt.subplots(1,1)
        kwargs.pop('title',None)
        field_image(fig,ax,display_field,**kwargs)
        ax.plot(rej_pos[1],rej_pos[0],'.',color='red',label='rejected objects')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1,1)
        kwargs.pop('title',None)
        field_image(fig,ax,field,**kwargs)
        ax.plot(rej_pos[1],rej_pos[0],'.',color='red',label='rejected objects')
        ax.plot(sel_pos[1],sel_pos[0],'.',color='blue',label='chosen objects')
        ax.legend()
        plt.show()

    # check routine finds at least one object
    if len(sel_obj[0]) == 0: return None
    
    print(f'\nRej:\t{len(rej_obj[0])}\nExt:\t{len(sel_obj[0])}')
    print(':: End ::')
    # extract list of objects and their std
    obj, err = sel_obj
    return obj, err, sel_pos


def new_kernel_fit(obj: NDArray, err: NDArray | None = None, initial_values: list[int] | None = None, display_fig: bool = False, **kwargs) -> tuple[NDArray,NDArray]:
    """To compute a gaussian fit for an object

    Parameters
    ----------
    obj : NDArray
        the object matrix
    err : NDArray | None, default None
        the STD of the object 
    initial_values : list[int] | None, default None
        values to initialize the fit process
    display_fig : bool, default False
        parameter to display the result of the fitting 
        procedure

    Returns
    -------
    pop : NDArray
        parameters from best fit
    Dpop : NDArray
        uncertainties on `pop`
    """
    ## Fit Function
    # define the fit function
    def fit_func(pos: tuple[NDArray,NDArray], k: float, sigma: float, x0: int, y0: int) -> NDArray:
        """Gaussian model for the fit

        Parameters
        ----------
        pos : tuple[NDArray,NDArray]
            x and y coordinates of each pixel of the object
        k : float
            normalization constant
        sigma : float
            variance root of the gaussian function
        x0 : int
            mean along x-axis
        y0 : int
            mean along y-axis

        Returns
        -------
        NDArray
            the value of the Gaussian
        """
        x, y = pos      #: coordinates of object pixels        
        # initialize the gaussian
        kernel = Gaussian(sigma)
        # compute the value
        return k * kernel.value(x-x0)*kernel.value(y-y0)

    xdim, ydim = obj.shape      #: sizes of the object
    xmax, ymax = peak_pos(obj)  #: coordinates of the brightest pixel

    ## Fit Pipeline
    # prepare data for the fit
    x = np.arange(xdim)
    y = np.arange(ydim) 
    y,x = np.meshgrid(y,x)      
    if initial_values is None:  #: default values to initialize the fit
        k0 = obj.max()      #: normalization constant
        # compute the initial value for sigma from HWHM
        hm = obj[xmax,ymax]/2   #: half maximum
        # find coordinates of the best estimation of `hm`
        hm_x, hm_y = np.unravel_index(abs(obj-hm).argmin(), obj.shape)
        # compute hwhm
        hwhm = np.sqrt((hm_x - xmax)**2 + (hm_y - ymax)**2)
        # compute sigma from it
        sigma0 = hwhm / (2*np.log(2))
        print('\nsigma0 =',sigma0)
        # collect initial values
        initial_values = [k0,sigma0,xmax,ymax]
    # adjust data shape for the fit
    xfit = np.vstack((x.ravel(),y.ravel()))
    yfit = obj.ravel()
    errfit = err.ravel() if err is not None else err
    # compute the fit
    fit = FuncFit(xdata=xfit, ydata=yfit, yerr=errfit)
    fit.pipeline(fit_func,initial_values,names=['k','sigma','x0','y0'])
    # store the results
    pop, Dpop = fit.results()    
    
    ## Plotting
    if display_fig:
        sigma, Dsigma = pop[1], Dpop[1]
        x0, y0 = pop[2:]
        figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else None
        title = kwargs['title'] if 'title' in kwargs.keys() else ''
        title = title + f' sigma0 = {sigma0:.2}\nsigma = {sigma:.2} +- {Dsigma:.2}'
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.set_title(title)
        field_image(fig,ax,obj)
        mx = np.linspace(-1,xdim,50)
        my = np.linspace(-1,ydim,50)
        yy, xx = np.meshgrid(my,mx)
        ax.contour(yy,xx,fit_func((xx,yy),*pop),colors='b',linestyles='dashed',alpha=0.7)
        ax.plot(ymax,xmax,'.r',label='Brightest pixel')
        ax.plot(y0,x0,'xb',label='Estimated mean')
        ax.set_xlim(0,obj.shape[1]-1)
        ax.set_ylim(0,obj.shape[0]-1)
        ax.legend()
        if err is not None:
            plt.figure()
            plt.subplot(1,2,1)
            plt.title('On x axis')
            plt.errorbar(np.arange(ydim),obj[xmax,:],yerr=err[xmax,:],fmt='.')
            plt.plot(my,fit_func([xmax,my],*pop))
            plt.subplot(1,2,2)
            plt.title('On y axis')
            plt.errorbar(np.arange(xdim),obj[:,ymax],yerr=err[:,ymax],fmt='.')
            plt.plot(mx,fit_func([mx,ymax],*pop))
        plt.show()
    return pop, Dpop

def kernel_estimation(extraction: list[NDArray], errors: list[NDArray], dim: int, selected: slice = slice(None), results: bool = True, display_plot: bool = False, **kwargs) -> tuple[float, float]:
    """To estimate the parameters of gaussian kernel

    Parameters
    ----------
    extraction : list[NDArray]
        list of extracted objects
    errors : list[NDArray]
        list of uncertainties of `extraction`
    dim : int
        size of the field matrix
    selected : slice, default slice(None)
        it is possible to select a certain number of object only over which the fit is computed
    display_plot : bool, default False
        parameter to display plots and figures

    Returns
    -------
    sigma : float
        computed mean from all fit estimations
    Dsigma : float
        computed STD from the mean `sigma`
    """
    # copy the lists to prevent errors
    sel_obj = [*extraction[selected]]
    sel_err = [*errors[selected]]
    a_sigma = np.empty(shape=(0,2),dtype=float) #: array to store values and uncertainties of sigma
    a_w = []

    ## Fit Routine
    for obj, err in zip(sel_obj, sel_err):
        # compute the fit
        pop, Dpop = new_kernel_fit(obj,err,initial_values=None,display_fig=display_plot,**kwargs)
        sigma = pop[1]
        Dsigma = Dpop[1]
        if sigma > 0:   #: check the goodness of the fit estimation
            # store the values
            a_sigma = np.append(a_sigma,[[sigma,Dsigma]],axis=0)
            a_w += [obj.max()]
        del sigma, Dsigma
    
    ## Estimation
    # check the presence of data
    if len(a_sigma) == 0: raise
    elif len(a_sigma) == 1: sigma, Dsigma = a_sigma[0]
    else:
        # computing the mean and STD
        sigma, Dsigma = mean_n_std(a_sigma[:,0],weights=a_w)
        # sigma, Dsigma = mean_n_std(a_sigma[:,0])
    print(f'\nsigma = {sigma:.5f} +- {Dsigma:.5f} -> {Dsigma/sigma*100:.2} %')
    
    ## Plotting
    # computing the kernel
    kernel = Gaussian(sigma).kernel()    
    if results:
        if 'title' not in kwargs:
            kwargs['title'] = f'Estimated kernel\n$\\sigma = $ {sigma:.2} $\pm$ {Dsigma:.2}'
        fast_image(kernel,**kwargs)
    
    return sigma, Dsigma


def LR_deconvolution(field: NDArray, kernel: DISTR, sigma: NDArray, mean_bkg: float, sigma_bkg: float, max_iter: int | None = None, display_fig: bool = False, **kwargs) -> NDArray:
    """To provide the Richardson-Lucy algorithm

    Parameters
    ----------
    field : NDArray
        the field matrix
    kernel : NDArray
        the estimated psf kernel
    mean_val : float
        the mean value of background
    sigma : NDArray
        the uncertainty of each pixel
    max_iter : int | None, default None 
        the algorithm is recursive, then one can set an iterations limit to avoid `StackOverflow Error`
    display_fig : bool, default False
        parameter to display plots and figures

    Returns
    -------
    rec_field : NDArray
        the field after the recover procedure
    
    Notes
    -----
    The algorithm (see [1]_ and [2]_) provides `I = S @ P + N`

        1. `I[r] = S[r] @ P`
        2. `S[r+1] = S[r] * (I/I[r] @ P)`

    `@` is the convolution operation

    The routine stops when:
        
        3. `| int I[r+1] - I[r] | < sqrt(Var(N))` 

    References
    ----------
    .. [1] L. B. Lucy, "An iterative technique for the rectification of observed distributions", 
        ApJ, 79:745, June 1974. doi: 10.1086/111605. 

    .. [2] W. H. Richardson, "Bayesian-based iterative method of image restoration", 
        Journal of the Optical Society of America (1917-1983), 62(1):55, January 1972. 
    """
    # import the package required to integrate and convolve
    from scipy.integrate import trapezoid
    from scipy.signal import convolve2d
    # from scipy.ndimage import convolve

    # def convolve(arr: NDArray, ker: NDArray) -> NDArray:
    #     bkg = Gaussian(sigma=sigma_bkg, mu=mean_bkg)
    #     pad_size = int(4*kernel.sigma)
    #     ext_arr = np.pad(arr,pad_size,mode='constant')
    #     mask = np.where(ext_arr == 0)
    #     pad_shape = ext_arr[mask].shape
    #     if len(pad_shape) == 1: 
    #         pad_shape = pad_shape[0] 
    #     ext_arr[mask] = bkg.field(pad_shape)
    #     conv_arr = convolve2d(ext_arr, ker, mode='same', boundary='fill', fillvalue=mean_bkg)
    #     edges = slice(pad_size, -pad_size)
    #     return conv_arr[edges, edges]

    ## Parameters    
    Dn = sigma.std()        #: the variance root of the uncertainties
    I = np.copy(field)     #: the field before recovery routine
    P = np.copy(kernel.kernel())     #: the estimated kernel
    bkg = Gaussian(sigma_bkg, mean_bkg)
    #> define the two recursive functions of the algorithm
    #.. they compute the value of `I` and `S` respectively at
    #.. the iteration r 
    # Ir = lambda S: convolve(S, P)
    # Sr = lambda S,Ir: S * convolve(I/Ir, P)
    Ir = lambda S: field_convolve(S, P, bkg)
    Sr = lambda S,Ir: S * field_convolve(I/Ir, P, bkg)

    ## Initialization
    r = 1               #: number of iteration
    Ir0 = Ir(I)         #: initial value for `I`
    # compute the first step
    Sr1 = Sr(I,Ir0)
    Ir1 = Ir(Sr1)
    # estimate the error
    diff = abs(trapezoid(trapezoid(Ir1-Ir0)))
    # print
    print('Dn', Dn)
    print(f'{r:03d}: - diff {diff}')
    while diff > Dn:
        r += 1
        # store the previous values
        Sr0 = Sr1
        Ir0 = Ir1
        # compute the next step
        Sr1 = Sr(Sr0,Ir0)
        Ir1 = Ir(Sr1)
        # estimate the error
        diff = abs(trapezoid(trapezoid(Ir1-Ir0)))
        print(f'{r:03d}: - diff {diff}')
        if max_iter is not None:    #: limit in iterations 
            if r > max_iter: 
                print(f'Routine stops due to the limit in iterations: {r} reached')
                break
    # store the result
    rec_field = Sr(Sr1,Ir1)

    ## Plotting
    if display_fig:
        fast_image(rec_field,**kwargs)
        fast_image(rec_field, norm='log',**kwargs)
    return rec_field

def mask_size(recover_field: NDArray, field: NDArray | None = None, display_fig: bool = False, **kwargs) -> int:
    tmp_field = recover_field.copy()
    dim = len(tmp_field)
    size = 0
    methods = lambda i : np.array([ 1 if tmp_field[i+1,i+1] - tmp_field[i,i] > 0 else 0,
                                    1 if tmp_field[dim-1-i-1,dim-1-i-1] - tmp_field[dim-1-i,dim-1-i] > 0 else 0,
                                    1 if tmp_field[i+1,dim-1-i-1] - tmp_field[i,dim-1-i] > 0 else 0,
                                    1 if tmp_field[dim-1-i-1,i+1] - tmp_field[dim-1-i,i] > 0 else 0
                                  ])
    i = 0
    diff = methods(i)
    while diff.sum() != 0: 
        diff[diff != 0] = methods(i)[diff != 0]
        i += 1
    size = i
    print('Lim Cut',size)
    size0 = size
    methods = lambda i : np.array([ tmp_field[size+1:,i+1] - tmp_field[size+1:,i],
                                    tmp_field[size+1:,dim-1-i-1] - tmp_field[size+1:,dim-1-i],
                                    tmp_field[i+1,size+1:] - tmp_field[i,size+1:],
                                    tmp_field[dim-1-i-1,size+1:] - tmp_field[dim-1-i,size+1:]
                                  ]).max()
    i = size + 1
    while methods(i) < 0: 
        i += 1
    size = i 
    # size = 4*i//3 
    print('Lim Cut',size)
    if display_fig:
        if field is not None:
            fig, (ax1,ax2) = plt.subplots(1,2)
            field_image(fig,ax1,field,colorbar=False)
            ax1.axhline(size0,0,1,color='red',label='size0')
            ax1.axhline(4*size0//3,0,1,color='orange',label='1/3')
            ax1.axhline(size,0,1,label='size')
            ax1.axhline(dim-1-size,0,1)
            ax1.axvline(size,0,1)
            ax1.axvline(dim-1-size,0,1)
            field_image(fig,ax2,tmp_field)
            ax2.axhline(size0,0,1,color='red',label='size0')
            ax2.axhline(4*size0//3,0,1,color='orange',label='1/3')
            ax2.axhline(size,0,1,label='size')
            ax2.axhline(dim-1-size,0,1)
            ax2.axvline(size,0,1)
            ax2.axvline(dim-1-size,0,1)
            fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
            field_image(fig,ax1,tmp_field,sct=((dim//2,None),(0,dim//2)),colorbar=False)
            ax1.axhline(dim//2-4*size0//3,0,1,color='orange',label='1/3')
            ax1.axvline(4*size0//3,0,1,color='orange',label='1/3')
            ax1.axhline(dim//2-size,0,1)
            ax1.axvline(size,0,1)
            field_image(fig,ax2,tmp_field,sct=(dim//2,None),colorbar=False)
            ax2.axhline(dim//2-4*size0//3,0,1,color='orange',label='1/3')
            ax2.axvline(dim//2-4*size0//3,0,1,color='orange',label='1/3')
            ax2.axhline(dim//2-size,0,1)
            ax2.axvline(dim//2-size,0,1)
            field_image(fig,ax3,tmp_field,sct=(0,dim//2),colorbar=False)
            ax3.axhline(4*size0//3,0,1,color='orange',label='1/3')
            ax3.axvline(4*size0//3,0,1,color='orange',label='1/3')
            ax3.axhline(size,0,1)
            ax3.axvline(size,0,1)
            field_image(fig,ax4,tmp_field,sct=((0,dim//2),(dim//2,None)),colorbar=False)
            ax4.axhline(4*size0//3,0,1,color='orange',label='1/3')
            ax4.axvline(dim//2-4*size0//3,0,1,color='orange',label='1/3')
            ax4.axhline(size,0,1)
            ax4.axvline(dim//2-size,0,1)
            plt.show()
        else:
            fig, ax = plt.subplots(1,1)
            field_image(fig,ax,tmp_field)
            ax.axhline(size0,0,1,color='red')
            ax.axhline(4*size0//3,0,1,color='orange')
            ax.axhline(size,0,1)
            ax.axhline(dim-1-size,0,1)
            ax.axvline(size,0,1)
            ax.axvline(dim-1-size,0,1)
            plt.show()
    return size

def mask_filter(field: NDArray, field0: NDArray | None = None, display_fig: bool = False,**kwargs) -> tuple[NDArray,int]:
    lim = mask_size(field,field0,display_fig=display_fig,**kwargs)
    mask = field.copy()
    dim = len(field)
    cut = slice(lim,dim-lim)
    mask[cut,cut] = 0.0
    if display_fig:
        fast_image(field-mask,**kwargs)
    # cut = slice(2*lim,dim - 2*lim)
    # amp, Damp = mean_n_std(field[cut,cut]/field0[cut,cut])
    # print(f'New Amp:\t{amp*100} +- {Damp*100} %')
    return field - mask, lim

def light_recover(obj: NDArray, a_size: NDArray[np.int64] | None = None, kernel: NDArray | None = None, mode: str = 'mean') -> tuple[float, float] | float:
    if mode == 'mean':
        xmax, ymax = peak_pos(kernel)
        xu, xd, yu, yd = a_size.flatten()
        xr = slice(xmax-xd,xmax+xu+1)
        yr = slice(ymax-yd,xmax+yu+1)
        cut = kernel[xr,yr].copy()
        l = obj/cut
        if len(l) == 1: 
            print(obj)
            fast_image(obj)
            raise
        L, DL = mean_n_std(l)
        return L, DL
    elif mode == 'integrate':
        from scipy.integrate import trapezoid,simpson
        L = trapezoid(trapezoid(obj))
        print(f'SIM = {simpson(simpson(obj))}')
        return L, L
    
def find_objects(field: NDArray, field0: NDArray, kernel: NDArray, mean_val: float, sel_pos: NDArray, size: int = 7, acc: float = 1e-5, mask: bool = True, method: str = 'integrate',res_str: str = ['lum','acc'], display_fig: bool = False, **kwargs) -> list[NDArray] | NDArray:
    dim = len(field)
    new_pos = sel_pos.copy()
    tmp_field = np.copy(field)                  #: field from which objects will be removed
    lum = np.empty(shape=(2,0),dtype=float)     #: array to collect brightness values and errors
    rej = []                                    #:
    rej_pos = np.empty(shape=(2,0),dtype=int)   #: array to collect rejected objects
    # applying the squared mask 
    if mask:
        tmp_field, lim = mask_filter(tmp_field,field0,display_fig=display_fig,**kwargs)
    # checking previous objects first
    for i in range(len(new_pos[0])):
        print('Iteration',i)
        x,y = new_pos[:,i]
        print(f'X,Y = {x,y}')
        xlim = (max(0,x-2),min(x+2,dim))
        ylim = (max(0,y-2),min(y+2,dim))
        print(xlim,ylim)
        xcut = slice(*xlim)
        ycut = slice(*ylim)
        x,y = np.array(peak_pos(tmp_field[xcut,ycut])) + np.array((xlim[0],ylim[0]))
        new_pos[:,i] = [x,y]
        print(f'X,Y new = {x,y}')
        if tmp_field[x,y]==0:
            rej += [i]
            rej_pos = np.append(rej_pos,[[x],[y]],axis=1) 
        else:
            a_size = new_grad_check(tmp_field,(x,y),mean_val,size=size)
            xu, xd, yu, yd = a_size.flatten()
            xr = slice(x-xd, x+xu+1) 
            yr = slice(y-yd, y+yu+1)
            obj = field[xr,yr].copy()
            # if obj is not acceptable
            if len(obj) <= 2: 
                rej += [i]
                rej_pos = np.append(rej_pos,[[x],[y]],axis=1) 
            else:
                # computing and storing brightness    
                l, Dl = light_recover(obj,a_size,kernel,mode=method)
                # l = field[x,y].copy()
                lum = np.append(lum,[[l],[Dl]],axis=1)
                if display_fig:
                    fast_image(obj,title=f'Obj n.{i} - ({x},{y})',**kwargs)        
            tmp_field[xr,yr] = 0.0
    pos = np.copy(new_pos)
    if len(rej) > 0:
        # removing rejected objects
        new_pos = np.delete(new_pos,rej,axis=1)
    if display_fig:
        fig, (ax1,ax2) = plt.subplots(1,2)
        fig.suptitle('New positions for previous objects')
        field_image(fig,ax1,field0,**kwargs)
        field_image(fig,ax2,field,**kwargs)
        if len(rej) > 0:
            ax1.plot(rej_pos[1],rej_pos[0],'x',color='red')
            ax2.plot(rej_pos[1],rej_pos[0],'.',color='red')
        ax1.plot(sel_pos[1],sel_pos[0],'.',color='orange')
        ax2.plot(new_pos[1],new_pos[0],'.',color='pink')
        plt.show()
    acc_pos = np.empty(shape=(2,0),dtype=int)      #: array to collect coordinates of stars
    unacc_pos = np.empty(shape=(2,0),dtype=int)
    # computing the amplification factor
    amp_fact = np.mean(field[pos[0],pos[1]]/field0[pos[0],pos[1]])
    print(f'Estimated Ampl Fact:\t{amp_fact*100} %')
    if amp_fact == 0: 
        print('rej',rej)
        print('pos',pos)
        raise
    # routine
    index = peak_pos(tmp_field)
    peak = tmp_field[index]
    while peak > mean_val:
        x,y = index
        pos = np.append(pos,[[x],[y]],axis=1)
        a_size = new_grad_check(tmp_field,index,mean_val,size=size)
        xu, xd, yu, yd = a_size.flatten()
        xr = slice(x-xd, x+xu+1) 
        yr = slice(y-yd, y+yu+1)
        amp = peak/field0[index]
        print(f'- Ampl Fact:\t{amp*100} %')
        obj = field[xr,yr].copy()      
        select = selection(obj,index,pos,size=size,mindist=0,minsize=2,sel=('size','dist'))
        edge_cond = sum([(np.array([x,y]) == i).any() for i in [lim+1,dim-2-lim]]) and np.logical_and(a_size < 3,a_size!=0).any() if mask else True
        if all(([[0,0],[0,0]] != a_size).any(axis=1)) and amp/amp_fact < 2 and select and edge_cond:
            l, Dl = light_recover(obj, a_size, kernel,mode=method)
            # l = field[x,y].copy()
            lum = np.append(lum,[[l],[Dl]],axis=1)
            acc_pos = np.append(pos,[[x],[y]],axis=1)
            if display_fig:
                fig, ax = plt.subplots(1,1)
                ax.set_title(f'Accepted {len(lum[0])} - ({x},{y})')
                field_image(fig,ax,obj,**kwargs)
                fig, (ax1,ax2) = plt.subplots(1,2)
                fig.suptitle(f'Accepted {len(lum[0])} - ({x},{y})')
                field_image(fig,ax1,field0,**kwargs)
                field_image(fig,ax2,field,**kwargs)
                ax1.plot(y,x,'.b')
                ax2.plot(y,x,'.b')                
                if len(rej) > 0:
                    ax1.plot(rej_pos[1],rej_pos[0],'x',color='violet')
                    ax2.plot(rej_pos[1],rej_pos[0],'.',color='violet')
                ax1.plot(sel_pos[1],sel_pos[0],'.',color='orange')
                ax2.plot(new_pos[1],new_pos[0],'.',color='orange')
        else:
            unacc_pos = np.append(unacc_pos,[[x],[y]],axis=1)
            if amp/amp_fact >= 2: cond = f'\namp/amp_factor = {amp/amp_fact}'
            else: cond = f'\na_size = {a_size}'; print(f'A_SIZE cond = {([[0,0],[0,0]] == a_size).any(axis=1)}\t{a_size}')
            if display_fig:
                fig, ax = plt.subplots(1,1)
                ax.set_title(f'Rejected - ({x},{y})'+cond)
                field_image(fig,ax,field[xr,yr],**kwargs)
                fig, (ax1,ax2) = plt.subplots(1,2)
                fig.suptitle(f'Rejected - ({x},{y})'+cond)
                field_image(fig,ax1,field0,**kwargs)
                field_image(fig,ax2,field,**kwargs)
                ax1.plot(y,x,'.r')
                ax2.plot(y,x,'.r')
                ax1.plot(sel_pos[1],sel_pos[0],'.',color='orange')
                ax2.plot(new_pos[1],new_pos[0],'.',color='orange')
                if len(rej) > 0:
                    ax1.plot(rej_pos[1],rej_pos[0],'x',color='violet')
                    ax2.plot(rej_pos[1],rej_pos[0],'.',color='violet')
        if display_fig and mask:
            ax2.axvline(lim,0,1,color='yellow',alpha=0.8)
            ax2.axhline(lim,0,1,color='yellow',alpha=0.8)
            ax2.axvline(dim-1-lim,0,1,color='yellow',alpha=0.8)
            ax2.axhline(dim-1-lim,0,1,color='yellow',alpha=0.8)
            plt.show()
        tmp_field[xr,yr] = 0.0
        index = peak_pos(tmp_field)
        peak = tmp_field[index]
    acc_pos = np.append(new_pos,acc_pos,axis=1)
    unacc_pos = np.append(rej_pos,unacc_pos,axis=1)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    field_image(fig,ax1,field0,**kwargs)
    field_image(fig,ax2,field,**kwargs)
    field_image(fig,ax3,tmp_field,**kwargs)
    ax2.plot(acc_pos[1],acc_pos[0],'.b')
    ax3.plot(acc_pos[1],acc_pos[0],'.b')
    ax3.plot(unacc_pos[1],unacc_pos[0],'.r')
    if mask:
        ax2.axvline(lim,0,1,color='yellow',alpha=0.8)
        ax2.axhline(lim,0,1,color='yellow',alpha=0.8)
        ax2.axvline(dim-1-lim,0,1,color='yellow',alpha=0.8)
        ax2.axhline(dim-1-lim,0,1,color='yellow',alpha=0.8)
        ax3.axvline(lim,0,1,color='yellow',alpha=0.8)
        ax3.axhline(lim,0,1,color='yellow',alpha=0.8)
        ax3.axvline(dim-1-lim,0,1,color='yellow',alpha=0.8)
        ax3.axhline(dim-1-lim,0,1,color='yellow',alpha=0.8)
    plt.show()
    results = []
    if res_str == 'all' or 'lum' in res_str:
        results += [lum]
    if res_str == 'all' or 'acc' in res_str:
        results += [acc_pos]
    if res_str == 'all' or 'rej' in res_str:
        results += [rej_pos]
    if res_str == 'all' or 'pos' in res_str:
        results += [pos]
    
    if len(results) == 0: raise Exception('Error in `res_str`')
    elif len(results) == 1: results = results[0]
    return results
