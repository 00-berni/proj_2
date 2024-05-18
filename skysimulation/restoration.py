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
    - [x] ~**Understand the squared artifact due to RL**~
          > They were convolution artifacts. I solved them padding the field before the routine
    - [] **Does padding the field before routine add some addition light?**
    - [] **Understand the width of the padding**
    - [] **Find a better way to extract objects**
          ? > A possible idea is to do a checking list:
              if peak/2 >= thr:
                -> Cut the image and study the gradient 
                -> Is the gradient ok? 
                  -| Yes, store the length
                  -| No
                   .> Check the brightness of the pixels
                   .> Is averaged px > bkg? 
                    -| Yes, store the length
                    -| No
                     .> Reject the object
              else:
                -> Study the gradient
                -> Cut the image
                -> Is the length comparable with mean sigma?
                  -| Yes, store the length
                  -| No
                   .> Reject the object




***
    
?WHAT ASK TO STEVE?
-------------------
    - [x] ***What's the role of binning? Is mine correct or not?***
          > Don't do the histogram, but a bar plot. You can do it also
            reducing data through grid (2x2 or 3x3 ...)
    - [x] *Is it good to use the brightness as a weight?*
          > Yes is a good choice
    - [] *Is it ok to use this method for the fit?*  
    - [x] ***Is the RL output good?***
          > No. You have to prevent the RL to touch the edge  
    - [x] ***After RL dec is taking the std of noise as error of each pixel good choice?***
          > (I_r - I_r-1) / (I_r + I_r-1)/2 
"""


from typing import Callable, Sequence, Literal, Any
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from .display import fast_image, field_image
from .stuff import Gaussian, DISTR
from .stuff import pad_field, field_convolve, mean_n_std, peak_pos, minimum_pos


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
        if sigma is not None or Dx is not None:
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
        if 'chisq' in self.res:
            chisq, chi0 = self.res['chisq']
            print(f'\tred_chi = {chisq/chi0*100:.2f} +- {np.sqrt(2/chi0)*100:.2f} %')

    def results(self) -> tuple[NDArray, NDArray] | tuple[None, None]:
        return self.fit_par, self.fit_err

    def pipeline(self,method: Callable[[Any,Any],Any], initial_values: Sequence[Any], names: list[str] | None = None,**kwargs) -> None:
        self.fit(method=method,initial_values=initial_values,**kwargs)
        self.infos(names=names)
    
    def gaussian_fit(self, intial_values: Sequence[Any], names: list[str] | None = None,**kwargs) -> None:
        """_summary_

        Parameters
        ----------
        intial_values : Sequence[Any]
            k, mu, sigma
        names : list[str] | None, optional
            names, by default None
        """
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





def bkg_est(field: NDArray, binning: int | Sequence[int | float] | None = None, display_plot: bool = False) -> tuple[tuple[float,float],float]:
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
        if ratio == 0: raise Exception("Binning is not possible")
        # set the number of bins
        binning = int(len(field) / np.log10(ratio)) *2 if ratio != 1 else int(len(field)*2)
        # print information
        print('\nBinning Results')
        print('Number of pixels:', data.shape[0])
        print('Magnitudes:', np.log10(ratio))
        print('Number of bins:', binning)
    
    ## Parameters Estimation
    # compute the histogram
    cnts, bins = np.histogram(data,bins=binning)
    # find the index of the max value
    max_indx = cnts.argmax()
    # compute the corresponding brightness value
    mean_bkg = (bins[max_indx+1] + bins[max_indx])/2
    Dmean_bkg = (bins[max_indx+1] - bins[max_indx])/2
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
    sigma_height = (cnts[max_indx] + cnts[sigma_indx])/2
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(cnts,height=sigma_height)
    # compute again the parameters in case of multiple peaks
    if len(peaks) > 1:
        # average between peaks
        mean_bkg, Dmean_bkg = mean_n_std(np.array([(bins[pk+1]+bins[pk])/2 for pk in peaks]))
        hm = cnts[peaks].mean()/2
        hm_indx = abs(cnts - hm).argmin()
        hwhm = abs((bins[hm_indx+1] + bins[hm_indx])/2 - mean_bkg)
        sigma_bkg = hwhm/np.sqrt(2*np.log(2))
        
    # print the results
    print('\nBackground estimation')
    print(f'\tmean:\t{mean_bkg} +- {Dmean_bkg}\n\tsigma:\t{sigma_bkg}\n\trelerr:\t{sigma_bkg/mean_bkg*100:.2f} %')
    
    ## Plotting
    if display_plot:
        # define variables to plot the estimated distribution
        #.. the distribution at higher values of brightness is affected by the 
        #.. brightness distribution of the stars, then only data within 3 sigma 
        #.. from the mean are taken 
        l_edge = abs(bins - (mean_bkg - 3*sigma_bkg)).argmin()      #: index of the bin at 3 sigma from mean
        l_val = (bins[l_edge + 1]+bins[l_edge])/2                   #: bin at 3 sigma from the mean
        xx = np.linspace(l_val, 2*mean_bkg - l_val ,len(data))
        model = Gaussian(sigma_bkg, mean_bkg)
        k = cnts[max_indx] / model.value((bins[max_indx+1] + bins[max_indx])/2 - mean_bkg)
        model = model.value(xx-mean_bkg) * k   
        
        plt.figure()
        plt.title(f'Gaussian Background Estimation\n mean = {mean_bkg:.4} +- {Dmean_bkg:.2} ; sigma = {sigma_bkg:.2}')
        plt.stairs(cnts, bins, fill=False, label='data')
        plt.axvline(mean_bkg, 0, 1, color='red', linestyle='dotted', label='estimated mean')
        plt.plot(xx,model, label='estimated gaussian')
        plt.plot((bins[peaks+1] + bins[peaks])/2,cnts[peaks],'.r')
        plt.axhline(sigma_height,0,1)
        plt.legend()
        plt.show()

    return (mean_bkg, Dmean_bkg), sigma_bkg

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
            if step == 0 or step <= back:
                # store the value
                if 'x' in direction and xd != 0: results  = [xsize] + results
                if 'y' in direction and yd != 0: results += [ysize]
                if len(results) == 1: results = results[0]
                return results
            elif step <= hm:
                # compute the gradient with the previous pixel
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


def selection(obj: NDArray, index: tuple[int,int], apos: NDArray, size: int, sel: Literal["all", "size", "dist", "new"] = 'all', mindist: int = 5, minsize: int = 3, debug_check: bool = False) -> bool:
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
    if sel == 'new':
        cond = True
        # xmax, ymax = peak_pos(obj)
        # dim = max([xmax, ymax, xdim-xmax, ydim-ymax])
        # mean_obj = np.array([[obj[xmax,ymax]],[obj[xmax,ymax]]])
        # for i in range(1,dim):
        #     xl = xmax - i
        #     xr = xdim - xmax - i
        #     yl = ymax - i
        #     yr = ydim - ymax - i
        #     mean_val_1 = []
        #     if xl >= 0: mean_val_1 += [obj[xl,ymax]]
        #     if xr >  0: mean_val_1 += [obj[xr,ymax]]
        #     if yl >= 0: mean_val_1 += [obj[xmax,yl]]
        #     if yr >  0: mean_val_1 += [obj[xmax,yr]]
        #     mean_val_1 = np.mean(mean_val_1)
        #     mean_val_2 = []
        #     if xl >= 0 and yl >= 0: mean_val_2 += [obj[xl,yl]]
        #     if xr >  0 and yl >= 0: mean_val_2 += [obj[xr,yl]]
        #     if xl >= 0 and yr >  0: mean_val_2 += [obj[xl,yr]]
        #     if xr >  0 and yr >  0: mean_val_2 += [obj[xr,yr]]
        #     mean_val_2 = np.mean(mean_val_2)
        #     mean_obj = np.append(mean_obj, [[mean_val_1], [mean_val_2]], axis=1)
        # grad1 = np.diff(mean_obj[0])
        # grad2 = np.diff(mean_obj[1])
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(mean_obj[0],'.--')
        # plt.plot(grad1,'+--')
        # plt.subplot(2,1,2)
        # plt.plot(mean_obj[1],'.--')
        # plt.plot(grad2,'+--')
        # fast_image(obj)
        # if len(np.where(grad1)[0]>0) != 0 and len(np.where(grad2)[0]>0) != 0:
        #     return False 
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
            rej_obj[0] += [field[xr,yr].copy()]
            rej_obj[1] += [sigma[xr,yr].copy()]
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
        values to initialize the fit process, `[k0, sigma0, x_mu, y_mu]`
    display_fig : bool, default False
        parameter to display the result of the fitting 
        procedure

    Returns
    -------
    pop : NDArray
        parameters from best fit, `[k, sigma, x_mu, y_mu]`
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
        k0 = obj[xmax,ymax]     #: normalization constant
        # compute the initial value for sigma from HWHM
        hm = k0/2   #: half maximum
        # find coordinates of the best estimation of `hm`
        hm_x, hm_y = minimum_pos(abs(obj-hm))        #np.unravel_index(abs(obj-hm).argmin(), obj.shape)
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

def kernel_estimation(extraction: list[NDArray], errors: list[NDArray], bkg: tuple[float,float], selected: slice = slice(None), results: bool = True, display_plot: bool = False, **kwargs) -> tuple[float, float]:
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
        # remove the contribution of the background
        m_bkg, Dm_bkg = bkg
        obj -= m_bkg
        err = np.sqrt(err**2 + Dm_bkg**2)
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
    # check the results
    if len(a_sigma) == 0: raise
    elif len(a_sigma) == 1: sigma, Dsigma = a_sigma[0]
    else:
        # computing the harmonic mean and STD
        s, Ds = mean_n_std(1/(a_sigma[:,0])**2,weights=a_w)
        sigma = np.sqrt(1/s)
        Dsigma = Ds/s * sigma / 2
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
    # import the package required to integrate
    from scipy.integrate import trapezoid

    ## Parameters    
    Dn = sigma.std()                        #: the variance root of the uncertainties
    P = np.copy(kernel.kernel())            #: the estimated kernel
    bkg = Gaussian(sigma_bkg, mean_bkg)     #: estimated background distribution
    # define the two recursive functions of the algorithm
    #.. they compute the value of `I` and `S` respectively at
    #.. the iteration r 
    Ir = lambda S: field_convolve(S, P, bkg, mode='2d')
    Sr = lambda S,Ir: S * field_convolve(I/Ir, P, bkg, mode='2d')
    # pad the field before convolutions
    #.. the field is put in a frame filled by drawing values 
    #.. from `bkg` distribution
    pad_size  = (len(P)-1)                  #: number of pixels to pad the field
    pad_slice = slice(pad_size,-pad_size)   #: field size cut
    I = pad_field(field, pad_size, bkg)
    
    ## RL Algorithm
    r = 1               #: number of iteration
    Ir0 = Ir(I)         #: initial value for I
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
    if display_fig:
        def sqr_mask(val: float, dim: int) -> NDArray:
            return np.array([ [val, val], 
                            [val, dim - val], 
                            [dim - val, dim - val], 
                            [dim - val, val],
                            [val, val] ])
        fig, ax = plt.subplots(1,1)
        ax.set_title('Before cutting')
        field_image(fig,ax,Sr(Sr1,Ir1))
        s_k = kernel.sigma
        mask0 = sqr_mask(s_k*1, len(I))
        mask1 = sqr_mask(s_k*2, len(I))
        mask2 = sqr_mask(s_k*3, len(I))
        mask3 = sqr_mask(s_k*4, len(I))
        ax.plot(mask0[:,1],mask0[:,0], color='blue')
        ax.plot(mask1[:,1],mask1[:,0], color='red')
        ax.plot(mask2[:,1],mask2[:,0], color='orange')
        ax.plot(mask3[:,1],mask3[:,0], color='green')
        plt.show()
    # store the result and remove the added frame
    rec_field = Sr(Sr1,Ir1)[pad_slice,pad_slice]

    ## Plotting
    if display_fig:
        fast_image(rec_field,'Recovered Field',**kwargs)
        fast_image(rec_field, norm='log',**kwargs)
    
    return rec_field

def light_recover(obj: NDArray, a_size: NDArray[np.int64] | None = None, kernel: NDArray | None = None, mode: str = 'mean') -> tuple[float, float] | float:
    from scipy.integrate import trapezoid
    L = trapezoid(trapezoid(obj))
    return L

def cutting(obj: NDArray, centre: Sequence[int]) -> tuple[NDArray, NDArray] | tuple[None, None]:
    WRONG_RESULT = None, None   #: the returned value for rejected objects
    xdim, ydim = obj.shape      #: sizes
    x0, y0 = centre             #: center coordinates
    hm = obj[x0, y0]/2          #: half maximum
    # compute the best approximation of hwhm
    hm_pos = max(minimum_pos(abs(obj-hm)))
    hwhm = np.rint((abs(hm_pos-x0) + abs(hm_pos-y0))/2).astype(int)
    if hwhm == 1:   #: check the size
        print('! HWHM nO')
        fast_image(obj,'! HWHM small')
        return WRONG_RESULT
    # select only the pixels inside hwhm
    cut = lambda centre, dim : slice(max(centre-hwhm,0), min(centre+hwhm +1 , dim))
    cut_obj = obj[cut(x0,xdim), cut(y0,ydim)].copy()
    if 1 in cut_obj.shape or 2 in cut_obj.shape:    #: check size
        fast_image(cut_obj,'! Sizes')
        print('! No shape')
        return WRONG_RESULT
    # xdim, ydim = cut_obj.shape
    print('\n\thwhm',hwhm)
    print('\thm_pos', hm_pos)
    print('\tsigma',hwhm / (2*np.log(2)))
    print('\tdim : ', xdim, ydim)
    print('\tcen : ', x0, y0)
    print('\tx : ', max(x0-hwhm,0), min(x0+hwhm+1, xdim))
    print('\ty : ', max(y0-hwhm,0), min(y0+hwhm+1, ydim))
    # compute the shift to trasform the coordinates
    xmax, ymax = peak_pos(obj)
    cxmax, cymax = peak_pos(cut_obj)
    shift = np.array([xmax - cxmax, ymax - cymax])
    print('shift',shift)
    return cut_obj, shift

def average_trend(obj: NDArray, centre: tuple[int, int]) -> tuple[NDArray, NDArray]:
    xdim, ydim = obj.shape
    val0 = obj[centre]
    x0, y0 = centre
    mean_obj = [[],[]]
    # compute pixel position
    step = lambda i, centre: centre + i
    # collect pixels along both horizontal and vertical directions
    mean_obj[0] += [ [obj[x,y0] for x in [step(i,x0), step(-i,x0)] if 0 <= x < xdim] + [obj[x0,y] for y in [step(i,y0), step(-i,y0)] if 0 <= y < ydim]  for i in range(1,max(xdim,ydim)) ]
    print('MEAN',mean_obj)
    # collect pixels along diagonal direction
    mean_obj[1] += [ [obj[x,y] for x in [step(i,x0), step(-i,x0)] for y in [step(i,y0), step(-i,y0)] if 0 <= x < xdim and 0 <= y < ydim] for i in range(1,max(xdim,ydim)) ]
    px1 = np.arange(1,len(mean_obj[0])+1)
    px2 = np.arange(1,len(mean_obj[1])+1)
    pos_px = np.argsort(np.append(px1,px2*np.sqrt(2)))
    px = np.append(px1,px2*-1)[pos_px]
    print(px)
    mm_obj = np.array([mean_obj[0][p-1] if p > 0 else mean_obj[1][-p-1] for p in px], dtype=object)
    mean_pos = [i for i in range(len(mm_obj)) if len(mm_obj[i]) != 0]
    mm_obj = np.array([np.mean(val) for val in mm_obj[mean_pos]])
    mm_obj = np.append([val0],mm_obj)
    px = np.where(px[mean_pos] < 0, px[mean_pos]*-np.sqrt(2), px[mean_pos])
    px = np.append([0],px)
    return px, mm_obj

        # grad1 = [[],[]]
        # grad2 = [[],[]]
        # mean_obj = [[],[]]
        # # compute pixel position
        # step = lambda i, centre: centre + i
        # # collect pixels along both horizontal and vertical directions
        # mean_obj[0] += [ [cut_obj[x,y0] for x in [step(i,x0), step(-i,x0)] if 0 <= x < xdim] + [cut_obj[x0,y] for y in [step(i,y0), step(-i,y0)] if 0 <= y < ydim]  for i in range(1,max(xdim,ydim)) ]
        # print('MEAN',mean_obj)
        # # collect pixels along diagonal direction
        # mean_obj[1] += [ [cut_obj[x,y] for x in [step(i,x0), step(-i,x0)] for y in [step(i,y0), step(-i,y0)] if 0 <= x < xdim and 0 <= y < ydim] for i in range(1,max(xdim,ydim)) ]
        # px1 = np.arange(1,len(mean_obj[0])+1)
        # px2 = np.arange(1,len(mean_obj[1])+1)
        # pos_px = np.argsort(np.append(px1,px2*np.sqrt(2)))
        # px = np.append(px1,px2*-1)[pos_px]
        # print(px)
        # mm_obj = np.array([mean_obj[0][p-1] if p > 0 else mean_obj[1][-p-1] for p in px], dtype=object)
        # mean_pos = [i for i in range(len(mm_obj)) if len(mm_obj[i]) != 0]
        # mm_obj = np.array([np.mean(val) for val in mm_obj[mean_pos]])
        # mm_obj = np.append([val0],mm_obj)
        # px = np.where(px[mean_pos] < 0, px[mean_pos]*-np.sqrt(2), px[mean_pos])
        # px = np.append([0],px)
        # print(px)
        # print('SHAPE',mm_obj.shape,px.shape)
        # fig0, ax0 = plt.subplots(1,1)
        # ax0.plot(px,mm_obj,'v-')
        # ax0.plot((px[1:]+px[:-1])/2,np.diff(mm_obj),'^--',color='orange',label='gradient')
        # ax0.axhline(0,0,1,color='black',alpha=0.5)
        # fig, ax = plt.subplots(2,1)
        # title = ''
        # for i in range(2):
        #     # average over pixels at same distance
        #     mean_obj[i] = np.array([np.mean(val) for val in mean_obj[i] if len(val) != 0])
        #     # append the centre pixel
        #     mean_obj[i] = np.append([val0],mean_obj[i])
        #     print(f'MEAN_{i}',mean_obj[i])
        #     if len(mean_obj[i]) > 1:
        #         # compute the first derivative
        #         grad1[i] = np.diff(mean_obj[i])
        #         ax[i].plot(grad1[i],'x--r')
        #         if len(grad1[i]) > 1:
        #             # compute the second derivative
        #             grad2[i] = np.diff(grad1[i])
        #             ax[i].plot(grad2[i],'+--g')
        #     pp = np.array([ np.where(mm_obj == m)[0][0] for m in mean_obj[i] if len(np.where(mm_obj == m)[0]) != 0 ])
        #     color = 'red' if i == 0 else 'green' 
        #     ax0.plot(px[pp],mm_obj[pp],marker='.',linestyle='--',color=color,label=f'obj {i}', alpha=0.5)

        #     ax[i].plot(mean_obj[i],'.--b')
        #     ax[i].axhline(0,0,1,color='black')
        #     title = title + i*'\n' + f'mean{i+1} = {np.mean(mean_obj[i][1:]):.2}, grad1_{i+1} = {np.mean(grad1[i]):.2}, grad2_{i+1} = {np.mean(grad2[i]):.2}'
        # #ax0.legend()
        # fig.suptitle(title)
        # fig, (ax1,ax2) = plt.subplots(1,2)
        # field_image(fig,ax1,obj)
        # field_image(fig,ax2,cut_obj)
        # ax1.plot(y0+shift[1],x0+shift[0],'.')
        # ax2.plot(y0,x0,'.')
        # plt.show()


def object_check(obj: NDArray, index: tuple[int,int], thr: float, sigma: Sequence[int] | None, mode: Literal["bright", "low"] = 'bright') -> tuple[NDArray, tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] | None:
    SIZE = 5
    xmax, ymax = peak_pos(obj)      #: max value coordinates
    xdim, ydim = obj.shape          #: sizes
    if mode == 'bright':
        ## Cut 
        print('\n\tFirst cut')
        # set the center
        centre = np.array([xmax, ymax])
        # select the pixels of interest from `obj`
        cut_obj, shift = cutting(obj, centre)
        if cut_obj is None:     #: check
            return None
        # fit with a gaussian in order to find the centroid
        pop, _ = new_kernel_fit(cut_obj-thr, display_fig=False)
        # check sigma
        est_sigma = pop[1]
        if est_sigma <= 0:  #: negative sigma is unacceptable
            print('BAD SIGMA')
            fast_image(cut_obj,'Bad Sigma')
            return None
        # check the centroid position
        if 0 <= pop[-2] < cut_obj.shape[0] and 0 <= pop[-1] < cut_obj.shape[1]:
            print('\n\tSecond cut')
            # compute center coordinates in the frame of `obj`
            centre = np.rint(pop[-2:]).astype(int) + shift 
            # select the pixels of interest from `obj`
            cut_obj, shift = cutting(obj, centre)
            if cut_obj is None:     #: check
                return None
        # change coordinates reference
        x0, y0 = centre - shift
        # compute the value of the centroid
        val0 = cut_obj[x0, y0]
        xdim, ydim = cut_obj.shape          #: sizes of the selected portion
        hwhm = (max(abs(xdim-x0),x0) + max(abs(ydim-y0),y0))/2
        sigma += [hwhm]

        ## Gradient
        print('cut_dim : ', xdim, ydim)
        # compute the mean trend
        px, mean_obj = average_trend(cut_obj, (x0,y0))
        # compute the first derivative
        grad1 = np.diff(mean_obj) 
        # check the derivative sign around the centre
        mean_width = np.rint(min(hwhm//2, 3)).astype(int)
        g_pos = np.where(grad1[:mean_width+1] >= 0)[0]
        fig0, ax0 = plt.subplots(1,1)
        ax0.set_title('Mean obj')
        ax0.plot(px,mean_obj,'.--',color='blue')
        m_px = (px[:-1] + px[1:])/2
        ax0.plot(m_px, grad1, 'v--', color='orange')
        ax0.plot((m_px[:-1]+m_px[1:])/2, np.diff(grad1), '^--', color='green')
        ax0.axhline(0,0,1,color='black')
        plt.show()
        if len(g_pos) == 0:
            # convert coordinates to the initial frame
            x0, y0 = np.array([x0, y0]) + shift
            # check the derivative sign out from the bulk
            g_pos = np.where(grad1[mean_width:] >= 0)[0]
            if len(g_pos) != 0:     #: cut the object
                xdim, ydim = obj.shape  #: initial sizes
                # cut at the first positive value of the derivative
                mean_width = g_pos[0]
                print('Quite')
                # compute the size of the object from the centre
                xsize = ( max(0, x0-mean_width),  min(xdim, x0+mean_width+1))
                ysize = ( max(0, y0-mean_width),  min(xdim, y0+mean_width+1))
                # cut the object
                obj = obj[slice(*xsize), slice(*ysize)]
                fast_image(obj,'cutted')
            print('GOOD')
        else:
            ## S/N
            # compute the ratio between pixels around the centre and mean background
            ratio = min(mean_obj[1:mean_width+1])/thr
            print('S/N',ratio*100,'%')
            plt.show()
            if ratio >= 0.5:
                print('Quite Quite Good')
                # convert coordinates to the initial frame
                x0, y0 = np.array([x0, y0]) + shift
                xdim, ydim = obj.shape  #: initial sizes
                # cut at the first positive value of the derivative
                mean_width = g_pos[0]
                print('Quite')
                # compute the size of the object from the centre
                xsize = ( max(0, x0-mean_width),  min(xdim, x0+mean_width+1))
                ysize = ( max(0, y0-mean_width),  min(xdim, y0+mean_width+1))
                # cut the object
                obj = obj[slice(*xsize), slice(*ysize)]
                fast_image(obj,'cutted S/N')
            else:
                print('RATIO',val0/thr*100,'%')
                print('NO GOOD')
                return None

    elif mode == 'low':
        fast_image(obj,'LOW before cutting')
        print('LOW')
        hwhm = np.rint(np.mean(sigma)).astype(int) if len(sigma) != 0 else SIZE
        cut = lambda centre, dim : slice(max(0, centre-hwhm), min(dim, centre + hwhm + 1))
        centre = np.array([xmax, ymax])
        x0, y0 = centre
        print(max(0, x0-hwhm), min(xdim, x0 + hwhm + 1))
        print(max(0, y0-hwhm), min(ydim, y0 + hwhm + 1))
        cut_obj = obj[cut(x0, xdim), cut(y0, ydim)].copy()
        fast_image(cut_obj,'cut_obj')
        cxmax, cymax = peak_pos(cut_obj)
        shift = np.array([xmax - cxmax, ymax - cymax])
        print('\n\thwhm', hwhm)
        print('\tsigma', hwhm / (2*np.log(2)))
        print('\tdim : ', xdim, ydim)
        print('\tcen : ', x0, y0)
        print('\tx : ', max(x0-hwhm,0), min(x0+hwhm+1, xdim))
        print('\ty : ', max(y0-hwhm,0), min(y0+hwhm+1, ydim))
        # fit with a gaussian in order to find the centroid
        pop, _ = new_kernel_fit(cut_obj-thr, display_fig=False)
        # check sigma
        est_sigma = pop[1]
        if est_sigma <= 0:  #: negative sigma is unacceptable
            print('BAD SIGMA')
            fast_image(cut_obj)
            return None
        # check the centroid position
        if 0 <= pop[-2] < cut_obj.shape[0] and 0 <= pop[-1] < cut_obj.shape[1]:
            centre = np.rint(pop[-2:]).astype(int) + shift
            x0, y0 = centre
            print(max(0, x0-hwhm), min(xdim, x0 + hwhm + 1))
            print(max(0, y0-hwhm), min(ydim, y0 + hwhm + 1))
            cut_obj = obj[cut(x0, xdim), cut(y0, ydim)].copy()
            fast_image(cut_obj,'Fit cut_obj')
            cxmax, cymax = peak_pos(cut_obj)
            shift = np.array([xmax - cxmax, ymax - cymax])
        fig, ax = plt.subplots(1,1)
        field_image(fig, ax, cut_obj)
        plt.show()
        # change coordinates reference
        x0, y0 = centre - shift
        val0 = cut_obj[x0,y0]
        xdim, ydim = cut_obj.shape
        px, mean_obj = average_trend(cut_obj, (x0,y0))
        grad1 = np.diff(mean_obj)
        fig0,ax0 = plt.subplots(1,1)
        ax0.set_title('Low: mean obj')
        ax0.plot(px,mean_obj,'.--',color='blue')
        m_px = (px[:-1] + px[1:])/2
        ax0.plot(m_px, grad1, 'v--', color='orange')
        ax0.plot((m_px[:-1]+m_px[1:])/2, np.diff(grad1), '^--', color='green')
        ax0.axhline(0,0,1,color='black')
        plt.show()
        ratio = min(mean_obj[1:])/thr
        if ratio >= 0.5:
            g_pos = np.where(grad1 >= 0)[0]
            if np.mean(grad1) < 0:#len(g_pos) == 0:
                obj = cut_obj
            else:
                print('gradient')
                print('NO GOOD')
                return None                
        else:
            print('RATIO',val0/thr*100,'%')
            print('NO GOOD')
            return None
    # compute the coordinates of the centre and the edges on the board
    #. the changing of coordinates is obtained from the
    #. maximum coordinates
    xdim, ydim = obj.shape
    x, y = index
    index = (x0 + (x-xmax), y0 + (y-ymax))
    xsize = (x-xmax, xdim + (x-xmax))
    ysize = (y-ymax, ydim + (y-ymax))
    return obj, index, (xsize, ysize)
                


            
def searching(field: NDArray, thr: float, errs: NDArray | None = None, max_size: int = 7, min_dist: int = 0, bright_cut: bool =  False, display_fig: bool = False, **kwargs) -> None | tuple[list[NDArray], NDArray]:
    def info_print(cnt: int, index: tuple, peak: float) -> None:
        x0 , y0 = index
        print(f'- - - -\nStep {cnt}')
        print(f'\tcoor : ({x0}, {y0})')
        print(f'\tpeak : {peak}')
    tmp_field = field.copy()
    display_field = field.copy()
    sigma = []
    arr_pos = np.empty((2,0),dtype=int)
    acc_obj = []
    acc_pos = np.empty((2,0),dtype=int)
    rej_obj = []
    rej_pos = np.empty((2,0),dtype=int)
    
    # first step
    xmax, ymax = peak_pos(tmp_field)    #: coordinates of the maximum
    peak = tmp_field[xmax, ymax]        #: maximum value
    stop_val = thr                      #: threashold
    cnt = 1
    print('\n- - - SEARCHING START - - -')
    print(f'Stop_val : {stop_val}')
    info_print(cnt,(xmax, ymax), peak)
    while peak > stop_val:
        # compute an estimation of the size of the object
        xsize, ysize = new_grad_check(tmp_field, (xmax, ymax), thr, size=max_size)
        # compute slices
        x = slice(xmax - xsize[0], xmax + xsize[1])
        y = slice(ymax - ysize[0], ymax + ysize[1])
        # define the object
        obj = field[x,y].copy()
        # remove small object
        if obj.shape[0] <= 3 or obj.shape[1] <= 3: 
                x0, y0 = xmax, ymax
                rej_obj += [obj]
                rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
        
        if peak/2 >= thr:
            fast_image(obj,'Object before check')
            print(f'\tshape : {obj.shape}')            
            fig0, ax0 = plt.subplots(1,1)
            field_image(fig0,ax0,display_field)
            ax0.plot(ymax,xmax,'.')
            # check if object is acceptable
            check = object_check(obj, (xmax, ymax), thr, sigma)
            if check is None:
                x0, y0 = xmax, ymax
                rej_obj += [obj]
                rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
            else:
                obj, (x0, y0), (xsize, ysize) = check
                print('xsize',xsize)
                print('ysize',ysize)
                # compute slices
                x = slice(*xsize)
                y = slice(*ysize)
                figg, axx = plt.subplots(1,2)
                field_image(figg,axx[0],obj)
                field_image(figg,axx[1],display_field[x,y])
                plt.show()
                # check 
                if selection(obj,(x0, y0), arr_pos, max_size,  mindist=min_dist, sel='all'):
                    acc_obj += [obj]
                    acc_pos = np.append(acc_pos, [[x0], [y0]], axis=1)
                    display_field[x,y] = 0.0
                else:
                    rej_obj += [obj]
                    rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
        else:
            fig0, ax0 = plt.subplots(1,1)
            field_image(fig0,ax0,display_field)
            ax0.plot(ymax,xmax,'.')
            # check if object is acceptable
            check = object_check(obj, (xmax, ymax), thr, sigma, mode='low')
            if check is None:
                x0, y0 = xmax, ymax
                rej_obj += [obj]
                rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
            else:
                obj, (x0, y0), (xsize, ysize) = check
                print('xsize',xsize)
                print('ysize',ysize)
                # compute slices
                x = slice(*xsize)
                y = slice(*ysize)
                figg, axx = plt.subplots(1,2)
                field_image(figg,axx[0],obj)
                field_image(figg,axx[1],display_field[x,y])
                plt.show()
                if selection(obj,(x0, y0), arr_pos, max_size,  mindist=min_dist, sel='all'):
                    acc_obj += [obj]
                    acc_pos = np.append(acc_pos, [[x0], [y0]], axis=1)
                    display_field[x,y] = 0.0
                else:
                    rej_obj += [obj]
                    rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
        # collect coordinates of the centre
        arr_pos = np.append(arr_pos, [[x0], [y0]], axis=1)
        # update the field
        tmp_field[x,y] = 0.0
        # compute the next step
        xmax, ymax = peak_pos(tmp_field)
        peak = tmp_field[xmax, ymax]   
        cnt += 1
        info_print(cnt,(xmax, ymax), peak)
    fig, ax = plt.subplots(1,1)
    field_image(fig,ax,display_field)
    if 0 not in acc_pos:
        ax.plot(acc_pos[1],acc_pos[0],'.b')
    if 0 not in rej_pos:
        ax.plot(rej_pos[1],rej_pos[0],'.r')
    plt.show()
    fig, ax = plt.subplots(1,1)
    field_image(fig,ax,field)
    if 0 not in acc_pos:
        ax.plot(acc_pos[1],acc_pos[0],'.b')
    plt.show()
    if 0 in acc_pos.shape: return None
    return acc_obj, acc_pos
    
