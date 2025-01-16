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
    !!- [] **Capire come mai un oggetto viene tagliato fino a farlo scomparire**

"""


from typing import Callable, Sequence, Literal, Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray, ArrayLike
from numpy import ndarray
from scipy.signal import find_peaks
from .display import fast_image, field_image
from .stuff import Gaussian, DISTR
from .stuff import distance, pad_field, field_convolve, mean_n_std, peak_pos, minimum_pos, unc_format, dist_corr

class StellarObject():

    def __init__(self, frame: NDArray, sigma: float, pos: tuple[int,int], errs: NDArray | None = None):
        self.frame = frame.copy()
        self.sigma = sigma
        self.errs  = errs.copy() if errs is not None else None
        self.pos   = (pos[0],pos[1])

    def max(self, **npargs) -> ArrayLike:
        return np.max(self.frame,**npargs)

    def update_errs(self,val:ArrayLike) -> 'StellarObject':
        if self.errs is not None:  
            new_obj = self.copy()
            new_obj.errs = np.sqrt(new_obj.errs**2 + val**2)

    def copy(self) -> 'StellarObject':
        return StellarObject(frame=self.frame,sigma=self.sigma, pos=self.pos,errs=self.errs)

    def __add__(self,val: ArrayLike) -> 'StellarObject':
        new_obj = self.copy()
        new_obj.frame += val
        return new_obj

    def __iadd__(self,val: ArrayLike) -> None:
        self = self + val

    def __sub__(self,val: ArrayLike) -> 'StellarObject':
        new_obj = self.copy()
        new_obj.frame -= val
        return new_obj

    def __isub__(self,val: ArrayLike) -> None:
        self = self - val

    def __mul__(self,val: ArrayLike) -> 'StellarObject':
        new_obj = self.copy()
        new_obj.frame *= val
        return new_obj
    def __truediv__(self,val: ArrayLike) -> 'StellarObject':
        new_obj = self.copy()
        new_obj.frame /= val
        return new_obj
    
    def __getitem__(self,index: int | tuple[int] | list[int] | slice ) -> NDArray:
        return self.frame[index]

    def __setitem__(self,index: int | tuple[int] | list[int] | slice , val: ArrayLike) -> NDArray:
        return self.frame.__setitem__(index,val)
    
class ListObjects():

    def __init__(self, input_list: list[StellarObject] = []):
        self.objs = [*input_list] 

    # def max(self) -> NDArray:

    # def copy(self) -> 'ListObjects':
    #     return ListObjects(self.objs)
    
    def __add__(self,obj: list[StellarObject] | StellarObject ) -> 'ListObjects':
        if isinstance(obj,StellarObject):
            return self.objs + [obj]
        elif isinstance(obj,ListObjects):
            return self.objs + obj.objs
        else:
            return self.objs + obj

    def __iadd__(self,obj: list[StellarObject] | StellarObject) -> None:
        self = self + obj
        
    def __getitem__(self, index: int | slice) -> StellarObject | list[StellarObject]:
        return self.objs[index]
    
    def __setitem__(self,index: int | slice, obj: StellarObject | list[StellarObject]) -> None:
        self.objs[index] = obj

class FuncFit():
    """To compute the fit procedure of some data

    Attributes
    ----------
    data : list[ndarray | None]
        the x and y data and (if there are) their uncertanties
    fit_par : ndarray | None
        fit estimated parameters
    fit_err : ndarray | None
        uncertanties of `fit_par`
    res : dict
        it collects all the results

    Examples
    --------
    Simple use:
    >>> def lin_fun(x,m,q):
    ...     return m*x + q
    >>> initial_values = [1,1]
    >>> fit = FuncFit(xdata=xfit, ydata=yfit, yerr=yerr)
    >>> fit.pipeline(lin_fun, initial_values, names=['m','q'])
    Fit results:
        m = 3 +- 1
        q = 0.31 +- 0.02
        red_chi = 80 +- 5 %
    >>> pop, Dpop = fit.results()

    A method provides the gaussian fit:
    >>> fit = FuncFit(xdata=xfit, ydata=yfit, yerr=errfit)
    >>> fit.gaussian_fit(initial_values, names=['k','mu','sigma'])
    Fit results:
        k = 10 +- 1
        mu = 0.01 +- 0.003
        sigma = 0.20 +- 0.01
        red_chi = 72 +- 15 %
    """
    @staticmethod
    def poly_func(xdata: ArrayLike, *pars) -> ArrayLike:
        ord = len(pars)-1
        poly = [ pars[i] * xdata**(ord - i) for i in range(ord+1)]
        return np.sum(poly,axis=0)

    @staticmethod
    def poly_error(xdata: ArrayLike, Dxdata: ArrayLike | None, par: ArrayLike, *errargs) -> ArrayLike:
        ord = len(par)-1
        err = 0
        if Dxdata is not None:
            err += (np.sum([ par[i] * (ord-i) * xdata**(ord-i-1) for i in range(ord)],axis=0) * Dxdata)**2
        if len(errargs) != 0:
            err += np.sum(np.square(errargs))
        err = np.sqrt(err)
        return err

    @staticmethod
    def gauss_func(xdata: ArrayLike, *args) -> ArrayLike:
        k, mu, sigma = args
        z = (xdata - mu) / sigma
        return k * np.exp(-z**2/2)

    @staticmethod
    def err_func(xdata: ArrayLike, Dxdata: ArrayLike | None, par: ArrayLike, *errargs) -> ArrayLike:
        coeff = FuncFit.gauss_func(xdata,*par) 
        err = 0
        if Dxdata is not None:
            err += (coeff * (xdata-par[1]) / par[2]**2 * Dxdata)**2
        if len(errargs) != 0:
            err += np.sum(np.square(errargs))
        err = np.sqrt(err)
        return err


    def __init__(self, xdata: ArrayLike, ydata: ArrayLike, yerr: ArrayLike | None = None, xerr: ArrayLike | None = None) -> None:
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
        if np.all(xerr == 0): xerr = None
        elif isinstance(xerr,(int,float)): xerr = np.full(xdata.shape,xerr)
        if np.all(yerr == 0): yerr = None
        elif isinstance(yerr,(int,float)): yerr = np.full(ydata.shape,yerr)
        xdata = np.copy(xdata)
        ydata = np.copy(ydata)
        xerr  = np.copy(xerr) if xerr is not None else None
        yerr  = np.copy(yerr) if yerr is not None else None
        self.data = [xdata, ydata, yerr, xerr]
        self.fit_par: ndarray | None = None
        self.fit_err: ndarray | None = None
        self.errvar: float | None = None
        self.res: dict = {}

    def odr_routine(self, **odrargs) -> None:
        xdata, ydata, yerr, xerr = self.data
        beta0 = self.res['init']
        from scipy import odr
        method = self.res['func']
        def fit_model(pars, x):
            return method(x, *pars)
        model = odr.Model(fit_model)
        data = odr.RealData(xdata,ydata,sx=xerr,sy=yerr)
        alg = odr.ODR(data, model, beta0=beta0,**odrargs)
        out = alg.run()
        pop = out.beta
        pcov = out.cov_beta
        Dpop = np.sqrt(pcov.diagonal())
        self.fit_par = pop
        self.fit_err = Dpop
        self.res['cov'] = pcov
        # if yerr is not None or xerr is not None:
        #     chisq = out.sum_square
        #     chi0 = len(ydata) - len(pop)
        #     self.res['chisq'] = (chisq, chi0)

    def chi_routine(self, err_func: Callable[[ArrayLike,ArrayLike,ArrayLike], ArrayLike] | None = None, iter: int = 3, **chiargs) -> None:
        xdata, ydata, yerr, xerr = self.data
        initial_values = self.res['init']
        method = self.res['func']
        from scipy.optimize import curve_fit
        if yerr is None: 
            chiargs['absolute_sigma'] = False
        pop, pcov = curve_fit(method, xdata, ydata, initial_values, sigma=yerr, **chiargs)
        sigma = yerr
        print('XERR',xerr)
        if xerr is not None:
            if yerr is None: yerr = 0
            initial_values = [*pop]
            sigma = np.sqrt(yerr**2 + err_func(xdata,xerr,pop))
            for _ in range(iter):
                pop, pcov = curve_fit(method, xdata, ydata, initial_values, sigma=sigma, **chiargs)
                initial_values = [*pop]
                sigma = np.sqrt(yerr**2 + err_func(xdata,xerr,pop))
        Dpop = np.sqrt(pcov.diagonal())
        self.fit_par = pop
        self.fit_err = Dpop
        self.res['cov'] = pcov
        # if sigma is not None:
        #     chisq = np.sum(((ydata - method(xdata,*pop)) / sigma)**2)
        #     chi0 = len(ydata) - len(pop)
        #     self.res['chisq'] = (chisq, chi0)        
            

    def fit(self, method: Callable[[Any,Any],Any], initial_values: Sequence[Any], mode: Literal['odr','curve_fit'] = 'curve_fit', **fitargs) -> None:
        """To compute the fit

        Parameters
        ----------
        method : Callable[[Any,Any],Any]
            the fit function
        initial_values : Sequence[Any]
            initial values
        """
        # importing the function
        # xdata, ydata = self.data[:2]
        # if len(xdata) != len(ydata): raise IndexError(f'different arrays length:\nxdata : {len(xdata)}\nydata : {len(ydata)}')
        self.res['func'] = method
        self.res['init'] = np.copy(initial_values)
        # if Dx is None: mode = 'curve_fit'
        self.res['mode'] = mode
        if mode == 'curve_fit':
            self.chi_routine(**fitargs)
        elif mode == 'odr':
            self.odr_routine(**fitargs)
        else: raise ValueError(f'mode = `{mode}` is not accepted')
        self.errvar = np.sqrt(np.var(self.residuals()))


    def infos(self, names: Sequence[str] | None = None) -> None:
        """To plot information about the fit

        Parameters
        ----------
        names : list[str] | None, default None
            list of fit parameters names
        """
        pop  = self.fit_par
        Dpop = self.fit_err
        print('\nFit results:')
        print('\tmode : '+self.res['mode'])
        initial_values = self.res['init']        
        if names is None:
            names = [f'par{i}' for i in range(len(pop))]
        for name, par, Dpar, init in zip(names,pop,Dpop,initial_values):
            fmt_m, fmt_u = unc_format(par,Dpar)
            fmt_measure = '\t{name}: {par:' + fmt_m[1:] + '} +/- {Dpar:' + fmt_u[1:] + '}  -->  {relerr:.2f} %\tinit : {init:.2}'
            # if Dpar != 0. : fmt_measure = fmt_measure + 
            try:
                info_str = fmt_measure.format(name=name, par=par, Dpar=Dpar,relerr=abs(Dpar/par)*100,init=init)
            except ValueError:
                info_str = f'\t{name}: {par} +/- {Dpar}  -->  {abs(Dpar/par)*100.0:.2f} %\tinit : {init*1.0:.2}'
            print(info_str)
        cov = self.res['cov']
        corr = np.array([ cov[i,j]/np.sqrt(cov[i,i]*cov[j,j]) for i in range(cov.shape[0]) for j in range(i+1,cov.shape[1])])
        names = np.array([ names[i] + '-' + names[j] for i in range(cov.shape[0]) for j in range(i+1,cov.shape[1])])
        for c, name in zip(corr,names):
            print(f'\tcorr_{name}\t = {c:.2}')
        if 'chisq' in self.res:
            chisq, chi0 = self.res['chisq']
            Dchi0 = np.sqrt(2*chi0)
            res = '"OK"' if abs(chisq-chi0) <= Dchi0 else '"NO"'
            if chi0 == 0:
                print('! ERROR !')
                print('\t',chisq,chi0)
                raise ValueError('Null degrees of freedom. Overfitting!')
            print(f'\tchi_sq = {chisq:.2f}  -->  red = {chisq/chi0*100:.2f} %')
            print(f'\tchi0 = {chi0:.2f} +/- {Dchi0:.2f}  -->  '+res)

    def results(self) -> tuple[ndarray, ndarray] | tuple[None, None]:
        return np.copy(self.fit_par), np.copy(self.fit_err)
    
    def method(self, xdata: ArrayLike) -> ArrayLike:
        func = self.res['func']
        pop = np.copy(self.fit_par)
        return func(xdata,*pop)

    def fvalues(self) -> ArrayLike:
        return self.method(self.data[0])
    
    def residuals(self) -> ArrayLike:
        return self.data[1] - self.fvalues()

    def pipeline(self,method: Callable[[Any,Any],Any], initial_values: Sequence[Any], names: Sequence[str] | None = None, mode: Literal['odr','curve_fit'] = 'curve_fit',**fitargs) -> None:
        self.fit(method=method,initial_values=initial_values,mode=mode,**fitargs)
        self.infos(names=names)
    
    def gaussian_fit(self, initial_values: Sequence[float] | None = None, names: Sequence[str] = ('k','mu','sigma'),mode: Literal['odr','curve_fit'] = 'curve_fit',**fitargs) -> None:
        """To fit with a Gaussian

        Parameters
        ----------
        intial_values : Sequence[Any]
            k, mu, sigma
        names : list[str] | None, optional
            names, by default None
        """
        if initial_values is None:
            xdata, ydata = self.data[0:2]
            maxpos = ydata.argmax()
            hm = ydata[maxpos]/2
            hm_pos = np.argmin(abs(hm-ydata))
            hwhm = abs(xdata[maxpos]-xdata[hm_pos])
            initial_values = [ydata[maxpos],maxpos,hwhm]

        if mode == 'curve_fit': 
            fitargs['err_func'] = FuncFit.err_func
        self.pipeline(method=FuncFit.gauss_func,initial_values=initial_values,names=names,mode=mode,**fitargs)
        
        # error_function = lambda x, Dx : FuncFit.err_func(x,Dx,self.fit_par,self.res['cov'])
        error_function = lambda x, Dx : FuncFit.err_func(x,Dx,self.fit_par,self.errvar)
        self.res['errfunc'] = error_function


    def pol_fit(self, ord: int | None = None, initial_values: Sequence[float] | None = None, names: Sequence[str] | None = None, mode: Literal['odr','curve_fit'] = 'curve_fit',**fitargs) -> None:
        if initial_values is None:
            if ord is None: raise ValueError('You have to set the grade of the polynomial')
            xdata, ydata = self.data[0:2]
            xtmp = np.copy(xdata)
            ytmp = np.copy(ydata)
            initial_values = [0]
            for _ in range(1,ord+1):
                ytmp = np.diff(ytmp)/np.diff(xtmp)
                initial_values += [np.mean(ytmp)]
                xtmp = (xtmp[1:] + xtmp[:-1])/2
            initial_values = initial_values[::-1] 
            initial_values[-1] = ydata[0] - FuncFit.poly_func(xdata[0],*initial_values)
            del xtmp,ytmp
        elif ord is None: ord = len(initial_values)-1
        if names is None:
            names = []
            for i in range(ord+1):
                names += [f'par_{ord-i}']
        if mode == 'curve_fit': 
            fitargs['err_func'] = FuncFit.poly_error
        self.pipeline(FuncFit.poly_func,initial_values=initial_values,names=names, mode=mode, **fitargs)
        print(self.res['cov'])
        self.res['errfunc'] = lambda x,Dx: FuncFit.poly_error(x,Dx,self.fit_par,self.errvar)
        # self.res['errfunc'] = lambda x,Dx: FuncFit.poly_error(x,Dx,self.fit_par)

    def linear_fit(self, initial_values: Sequence[float] | None = None, names: Sequence[str] = ('m','q'), mode: Literal['odr','curve_fit'] = 'odr',**fitargs) -> None:
        self.pol_fit(ord=1, initial_values=initial_values, names=names, mode=mode, **fitargs)
    
    def sigma(self) -> ArrayLike:
        err_func = self.res['errfunc']
        xdata, _, Dy, Dx = self.data
        err = err_func(xdata,Dx)
        if Dy is not None:
            err = np.sqrt(Dy**2 + err**2)
        return err

    def data_plot(self, ax: Axes, points_num: int = 200, grid: bool = True, pltarg1: dict = {}, pltarg2: dict = {},**pltargs) -> None:
        if 'title' not in pltargs.keys():
            pltargs['title'] = 'Fit of the data'
        if 'fontsize' not in pltargs.keys():
            pltargs['fontsize'] = 18
        if 'xlabel' not in pltargs.keys():
            pltargs['xlabel'] = ''
        if 'ylabel' not in pltargs.keys():
            pltargs['ylabel'] = ''
        title = pltargs['title']
        fontsize = pltargs['fontsize']
        ylabel = pltargs['ylabel']
        xlabel = pltargs['xlabel']
        xdata = self.data[0]
        xx = np.linspace(xdata.min(),xdata.max(),points_num)
        if 'fmt' not in pltarg1.keys():
            pltarg1['fmt'] = '.'        
        ax.errorbar(*self.data,**pltarg1)
        ax.plot(xx,self.method(xx),**pltarg2)
        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        ax.set_title(title,fontsize=fontsize+2)
        if grid: ax.grid(color='lightgray', ls='dashed')
        if 'label' in pltarg1.keys() or 'label' in pltarg2.keys():
            ax.legend(fontsize=fontsize)

    def residuals_plot(self, ax: Axes, grid: bool = True, **pltargs) -> None:
        if 'title' not in pltargs.keys():
            pltargs['title'] = 'Residuals of the data'
        if 'fontsize' not in pltargs.keys():
            pltargs['fontsize'] = 18
        if 'xlabel' not in pltargs.keys():
            pltargs['xlabel'] = ''
        if 'ylabel' not in pltargs.keys():
            pltargs['ylabel'] = 'residuals'
        title = pltargs['title']
        fontsize = pltargs['fontsize']
        ylabel = pltargs['ylabel']
        xlabel = pltargs['xlabel']
        pltargs.pop('title')
        pltargs.pop('fontsize')
        pltargs.pop('ylabel')
        pltargs.pop('xlabel')
        if 'fmt' not in pltargs.keys():
            pltargs['fmt'] = 'o'
        if 'linestyle' not in pltargs.keys():
            pltargs['linestyle'] = 'dashed'
        if 'capsize' not in pltargs.keys():
            pltargs['capsize'] = 3
        xdata = self.data[0]
        ax.errorbar(xdata,self.residuals(),self.sigma(),**pltargs)
        ax.axhline(0,0,1,color='black')
        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)
        ax.set_title(title,fontsize=fontsize+2)
        if grid: ax.grid(color='lightgray', ls='dashed')
        if 'label' in pltargs.keys():
            ax.legend(fontsize=fontsize)

    def plot(self, sel: Literal['data','residuals','all'] = 'all', mode: Literal['plots', 'subplots'] = 'plots', points_num: int = 200, fig: None | Figure | tuple[Figure,Figure] = None, grid: bool = True, plot1: dict = {}, plot2: dict = {}, plotargs: dict = {}, **resargs) -> None:
        if fig is None:
            if sel in ['data','all']: 
                fig = plt.figure()
            if sel in ['residuals', 'all'] and mode != 'subplots': 
                fig2 = plt.figure()
        elif isinstance(fig,tuple):
            fig, fig2 = fig
        if sel == 'all' and mode == 'subplots':
            ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
            plotargs['xlabel'] = ''
            resargs['title'] = ''        
            self.data_plot(ax=ax1,points_num=points_num,grid=grid,pltarg1=plot1,pltarg2=plot2,**plotargs)
            self.residuals_plot(ax=ax2,**resargs)
        else:
            if sel in ['data','all']:
                ax = fig.add_subplot(111)
                self.data_plot(ax=ax, points_num=points_num,grid=grid,pltarg1=plot1,pltarg2=plot2,**plotargs)
            if sel in ['residuals', 'all']:
                ax2 = fig2.add_subplot(111)
                self.residuals_plot(ax=ax2,**resargs)


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





def bkg_est(field: NDArray, binning: int | Sequence[int | float] | None = None, display_plot: bool = False,**pltargs) -> tuple[tuple[float,float],float]:
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
    if 'fontsize' not in pltargs.keys():
        pltargs['fontsize'] = 18    
    mean_bkg = np.median(field)
    window = 2*mean_bkg - field.min()
    sigma_bkg = np.sqrt(np.var(field[field<=window]))
    # print the results
    print('\nBackground estimation')
    print(f'\tmean:\t{mean_bkg}\n\tsigma:\t{sigma_bkg}\n\trelerr:\t{sigma_bkg/mean_bkg*100:.2f} %')
    
    ## Plotting
    if display_plot:
        frame = field.copy()        #: copy of the field matrix
        data = frame.flatten()      #: 1-D data array
        # if binning is None:
        #     # compute the magnitude between max and min data
        #     ratio = int(data.max()/data.min())
        #     if ratio == 0: raise Exception("Binning is not possible")
        #     # set the number of bins
        #     binning = int(len(field) / np.log10(ratio)) *2 if ratio != 1 else int(len(field)*2)
        #     # print information
        #     print('\nBinning Results')
        #     print('Number of pixels:', data.shape[0])
        #     print('Magnitudes:', np.log10(ratio))
        #     print('Number of bins:', binning)
        # # compute the histogram
        # cnts, bins = np.histogram(data,bins=binning)


        bkg_fmt = unc_format(mean_bkg,sigma_bkg)
        mean_label  = '$\\bar{n}_B$ = {mean:' + bkg_fmt[0][1:] + '}'
        sigma_label = '$\\sigma_B$ = {sigma:' + bkg_fmt[1][1:] + '}'
        plt.figure()
        plt.title(f'Background Estimation',fontsize=pltargs['fontsize']+2)
        # plt.stairs(cnts, bins, fill=False)
        plt.hist(data,len(field)*2,histtype='step')
        plt.axvline(mean_bkg, 0, 1, color='red', linestyle='dotted', label=mean_label.format(n='{n}',mean=mean_bkg))
        plt.axvspan(mean_bkg-sigma_bkg,mean_bkg+sigma_bkg, 0, 1, facecolor='orange', alpha=0.4, label=sigma_label.format(sigma=sigma_bkg))
        plt.xscale('log')
        plt.xlabel('$\\ell$ [a.u.]',fontsize=pltargs['fontsize'])
        plt.ylabel('counts',fontsize=pltargs['fontsize'])
        plt.legend(fontsize=pltargs['fontsize'])
        plt.show()

    return (mean_bkg, sigma_bkg), sigma_bkg

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
    n_mov = lambda val : new_moving(val,field,index,back,size,debug_check=debug_check)
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
    xsize, ysize = a_xysize.reshape(2,2)
    dim = len(field)
    x0, y0 = index
    if x0 - xsize[0] < 0: xsize[0] = x0
    if x0 + xsize[1] >= dim: xsize[1] = dim - x0 - 1   
    if y0 - ysize[0] < 0: ysize[0] = y0
    if y0 + ysize[1] >= dim: xsize[1] = dim - y0 - 1   
    
    ## Check
    tmp_obj = field[slice(x0-xsize[0],x0+xsize[1]+1),slice(y0-ysize[0],y0+ysize[1]+1)].copy()
    cxmax,cymax = peak_pos(tmp_obj)
    # if x0 == 33:
    #     print('Lims',cxmax,xsize[0])
    #     print('Lims',cymax,ysize[0])
    #     plt.figure()
    #     plt.imshow(tmp_obj)
    #     plt.plot(cymax,cxmax,'.')
    #     plt.plot(ysize[0],xsize[0])
    #     plt.show()
    if tmp_obj[cxmax,cymax] != field[x0,y0]:
        plt.figure()
        plt.imshow(tmp_obj)
        plt.plot(cymax,cxmax,'.')
        plt.figure()
        plt.imshow(field)
        plt.plot(y0,x0,'.')
        plt.show()
        raise
    if cxmax != xsize[0] or cymax != ysize[0]:
        print('oooooh')
        plt.figure()
        plt.imshow(tmp_obj)
        plt.plot(cymax,cxmax,'.')
        plt.plot(ysize[0],xsize[0])
        plt.show()
        print(cxmax,xsize[0])
        print(cymax,ysize[0])
        xsize[0] = cxmax
        ysize[0] = cymax
        raise
    return xsize, ysize


def selection(obj: NDArray, index: tuple[int,int], apos: NDArray, size: int, sel: Literal["all", "size", "dist", "new"] = 'all', mindist: int = 0, minsize: int = 3, debug_check: bool = False) -> bool:
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
# def object_isolation(field: NDArray, thr: float, sigma: NDArray, size: int = 5, objnum: int = 10, reshape: bool = False, reshape_corr: bool = False, sel_cond: bool = False, mindist: int = 5, minsize: int = 3, cutsize: int = 5, numpeaks: int = 2, grad_new: bool = True, corr_cond: bool = True, debug_check: bool = False, results: bool = True, display_fig: bool = False,**kwargs) -> tuple[list[NDArray], list[NDArray], NDArray] | None:
#     """To isolate the most luminous star object.
   
#     The function calls the `size_est()` function to compute the size of the object and
#     then to extract it from the field.


#     Parameters
#     ----------
#     field : NDArray
#         the field matrix
#     thr : float
#         value below which searching routine stops, 
#         e.g. the average value of the background
#     sigma : NDArray
#         STD of field pixels
#     size : int, default 5
#         maximum size of an extracted object
#     objnum : int, default 10
#         maximum number of object to extract
#     reshape : bool, optional
#         _description_, by default False
#     reshape_corr : bool, optional
#         _description_, by default False
#     sel_cond : bool, default False
#         _description_
#     mindist : int, default 5
#         _description_ 
#     minsize : int, default 3
#         _description_
#     cutsize : int, optional
#         _description_, by default 5
#     numpeaks : int, optional
#         _description_, by default 2
#     grad_new : bool, optional
#         _description_, by default True
#     corr_cond : bool, optional
#         _description_, by default True
#     debug_check : bool, optional
#         _description_, by default False
#     display_fig : bool, default False
#         parameter to show figures

#     Returns
#     -------
#     obj : list[NDArray]
#         list of extracted objects    
#     err : list[NDArray]
#         list of STD matrix for each 
#         extracted object
#     pos : NDArray
#         array of the coordinates of
#         all objects
#         The format is 
#             `[ [x_array], [y_array]  ]`
#     """
#     # make a copy of the field
#     tmp_field = field.copy()        #: field from which objects will be removed
#     display_field = field.copy()    #: field from which only selected objects will be removed (for plotting only)
#     # initialize variables to collect data
#     a_pos = np.empty(shape=(2,0),dtype=int)         #: array to store coordinates of all objects
#     sel_obj = [[],[]]                               #: list to collect accepted objects 
#     sel_pos = np.empty(shape=(2,0),dtype=int)       #: array to store coordinates of `sel_obj`
#     rej_obj = [[],[]]                               #: list to collect rejected objects
#     rej_pos = np.empty(shape=(2,0),dtype=int)       #: array to store coordinates of `rej_obj`
#     if display_fig:
#         tmp_kwargs = {key: kwargs[key] for key in kwargs.keys() - {'title'}} 
    
#     ## Extraction Routine
#     k = 0           #: counter of selected objects
#     ctrl_cnt = 0    #: counter of iterations
#     print('\n- - - Object Extraction - - -')
#     while k < objnum:
#         # find the position of the brightest pixel
#         index = peak_pos(tmp_field)
        
#         #? per me
#         if debug_check:
#             ctrl = False    #?: solo per me
#             if 0 in index: 
#                 print(index)
#                 ctrl = True
#         #?
        
#         # store the brightest pixel value
#         peak = tmp_field[index]
#         if peak <= thr:     #: stopping condition
#             break
#         # compute the size
#         a_size = new_grad_check(tmp_field,index,thr,size)
        
#         #? per me
#         if debug_check:
#             if ctrl: 
#                 print(index)
#                 print(a_size)
#         #?
                
#         print(f':: Iteration {k} of object_isolation :: ')
#         print('a_size',a_size)
#         x, y = index                        #: coordinates of the brightest pixel
#         xu, xd, yu, yd = a_size.flatten()   #: edges of the object
#         # compute slices for edges
#         xr = slice(x-xd, x+xu+1) 
#         yr = slice(y-yd, y+yu+1)
#         # remove the object from the field
#         tmp_field[xr,yr] = 0.0
#         if debug_check:
#             print('Slices: ',xr,yr)
#             print('a_size 2',a_size)
#         # update the array for coordinates
#         a_pos = np.append(a_pos,[[x],[y]],axis=1)
#         if any(([[0,0],[0,0]] == a_size).all(axis=1)):  #: a single pixel is not accepted
#             # storing the object and its coordinates
#             rej_obj[0] += [field[xr,yr].copy()]
#             rej_obj[1] += [sigma[xr,yr].copy()]
#             rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
#             if debug_check:
#                 print(f'Rejected obj: ({x},{y})')
#         else:            
#             # store the object and its std
#             obj = field[xr,yr].copy() 
#             err = sigma[xr,yr].copy()
#             # check the condition to accept an object
#             save_cond = selection(obj,index,a_pos,size,sel='all',mindist=mindist,minsize=minsize) if sel_cond else True
#             if save_cond:       #: accepted object
#                 if debug_check:
#                     print(f'** OBJECT SELECTED ->\t{k}')
#                 # remove it from the field for displaying
#                 display_field[xr,yr] = 0.0
#                 # store the selected object, its std and coordinates
#                 sel_obj[0] += [obj]
#                 sel_obj[1] += [err]
#                 sel_pos = np.append(sel_pos,[[x],[y]],axis=1)
#                 if display_fig:
#                     tmp_kwargs['title'] = f'N. {k+1} object {index} - {ctrl_cnt}'
#                 # update the counter of object
#                 k += 1 
#             else: 
#                 if debug_check:
#                     print(f'!! OBJECT REJECTED ->\t{k}')
#                 # store the rejected object, its std and coordinates
#                 rej_obj[0] += [obj]
#                 rej_obj[1] += [err]
#                 rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
#                 if display_fig:
#                     tmp_kwargs['title'] = f'Rejected object {index} - {ctrl_cnt}'
#             if display_fig:     #: show the object
#                 fig, ax = plt.subplots(1,1)
#                 field_image(fig,ax,tmp_field,**{key: tmp_kwargs[key] for key in tmp_kwargs.keys() - {'title'}})
#                 if len(rej_pos[0])!= 0:
#                     ax.plot(rej_pos[1,:-1],rej_pos[0,:-1],'.r')
#                     ax.plot(rej_pos[1,-1],rej_pos[0,-1],'x',color='red')
#                 if len(sel_pos[0])!=0:
#                     ax.plot(sel_pos[1,:-1],sel_pos[0,:-1],'.b')
#                     ax.plot(sel_pos[1,-1],sel_pos[0,-1],'xb')
#                 fast_image(obj,**tmp_kwargs) 
#             # check the condition to prevent the stack-overflow
#             if (ctrl_cnt >= objnum and len(sel_obj[0]) >= 3) or ctrl_cnt > 2*objnum and sel_cond:
#                 break
#             # update the counter of iterations
#             ctrl_cnt += 1

#     # display the field after extraction
#     if results:
#         if 'title' not in kwargs:
#             kwargs['title'] = 'Field after extraction'
#         fast_image(display_field,**kwargs)
#         fast_image(tmp_field,**kwargs)    
#     # show the field with markers for rejected and selected objects
#     if sel_cond and len(sel_obj[0]) > 0 and results:
#         fig, ax = plt.subplots(1,1)
#         kwargs.pop('title',None)
#         field_image(fig,ax,display_field,**kwargs)
#         ax.plot(rej_pos[1],rej_pos[0],'.',color='red',label='rejected objects')
#         ax.legend()
#         plt.show()
#         fig, ax = plt.subplots(1,1)
#         kwargs.pop('title',None)
#         field_image(fig,ax,field,**kwargs)
#         ax.plot(rej_pos[1],rej_pos[0],'.',color='red',label='rejected objects')
#         ax.plot(sel_pos[1],sel_pos[0],'.',color='blue',label='chosen objects')
#         ax.legend()
#         plt.show()

#     # check routine finds at least one object
#     if len(sel_obj[0]) == 0: return None
    
#     print(f'\nRej:\t{len(rej_obj[0])}\nExt:\t{len(sel_obj[0])}')
#     print(':: End ::')
#     # extract list of objects and their std
#     obj, err = sel_obj
#     return obj, err, sel_pos


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
        hm_x, hm_y = minimum_pos(abs(obj-hm))      
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
    try:
        fit.pipeline(fit_func,initial_values,names=['k','sigma','x0','y0'],mode='curve_fit')
    except RuntimeError: 
        return None, None
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
    sel_obj = [ np.copy(elem) for elem in extraction[selected]]
    sel_err = [ np.copy(elem) for elem in errors[selected]] if errors is not None else [1e-1000 * ext for ext in sel_obj]
    a_sigma = np.empty(shape=(0,2),dtype=float) #: array to store values and uncertainties of sigma
    a_w = []

    ## Fit Routine
    for obj, err in zip(sel_obj, sel_err):
        # remove the contribution of the background
        m_bkg, Dm_bkg = bkg
        # obj -= m_bkg
        # err = np.sqrt(err**2 + Dm_bkg**2)
        # compute the fit
        pop, Dpop = new_kernel_fit(obj,err,initial_values=None, display_fig=display_plot,**kwargs)
        if pop is not None:
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
            value_fmt = unc_format(sigma,Dsigma)
            value = '$\\sigma = ${sigma:' + value_fmt[0][1:] + '} $\\pm$ {Dsigma:' + value_fmt[1][1:] + '}'
            kwargs['title'] = 'Estimated kernel\n' + value.format(sigma=sigma,Dsigma=Dsigma)
        elif kwargs['title'] == 'title-only':
            kwargs['title'] = 'Estimated kernel'

        fast_image(kernel,**kwargs)
    
    return sigma, Dsigma


def LR_deconvolution(field: NDArray, kernel: DISTR, sigma: NDArray, mean_bkg: float, sigma_bkg: float, max_iter: int | None = None, max_r: int = 3000, thr_mul: float | None = None, mode: None | dict = None, display_fig: bool = False, **kwargs) -> NDArray:
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

    ## Parameters
    if thr_mul is None:
        thr_mul = 1/field.sum()    
    Dn = sigma.std()                        #: the variance root of the uncertainties
    # print('Dn', Dn)
    # Dn = sigma.mean()                        #: the variance root of the uncertainties
    # # Dn = sigma_bkg                        #: the variance root of the uncertainties
    P = np.copy(kernel.kernel())            #: the estimated kernel
    # define the two recursive functions of the algorithm
    #.. they compute the value of `I` and `S` respectively at
    #.. the iteration r 
    if mode is None:
        mode = {'boundary': 'fill', 'fillvalue': mean_bkg}
    from scipy.signal import convolve2d    
    Ir = lambda S: convolve2d(S, P,  mode='same', **mode)
    Sr = lambda S,Ir: S * convolve2d(I/Ir, P,  mode='same', **mode)
    # pad the field before convolutions
    #.. the field is put in a frame filled by drawing values 
    #.. from `bkg` distribution
    pad_size  = (len(P)-1)+2                  #: number of pixels to pad the field
    # pad_slice = slice(pad_size,-pad_size)   #: field size cut
    I = pad_field(field, pad_size, mean_bkg)
    # I = field.copy()
    
    ## RL Algorithm
    import time
    r = 1               #: number of iteration
    start_time = time.time()
    Ir0 = Ir(I)         #: initial value for I
    # compute the first step
    Sr1 = Sr(I,Ir0)
    Ir1 = Ir(Sr1)
    Ir1 = np.where(Ir1<0,0,Ir1)
    # estimate the error
    diff = np.abs(Ir1-Ir0).sum()/np.sum(I)
    chisq = ((I-Ir1)**2).sum()/(len(I)**2-1)
    # print
    print('Dn', Dn)
    print('Dn', Dn/I.sum())
    print('Dn',Dn*thr_mul)
    print(f'{r:04d}: - diff {diff:.3e}\tchisq {chisq:.3e}',end='\r')
    a_diff = []
    a_chisq = []
    stop_cond = r<max_r if Dn == 0 else diff >= Dn*thr_mul
    while stop_cond: #r<3000: #diff >= Dn:
        # if len(np.where(Ir1<=0)[0]):
        #     print(Ir1[Ir1<=0])
        #     exit()
        r += 1
        # store the previous values
        Sr0 = Sr1
        Ir0 = Ir1
        # compute the next step
        Sr1 = Sr(Sr0,Ir0)
        Ir1 = Ir(Sr1)
        Ir1 = np.where(Ir1<0,0,Ir1)
        # estimate the error
        diff = np.abs(Ir1-Ir0).sum()/np.sum(I)
        chisq = ((I-Ir1)**2).sum()/(len(I)**2-1)
        a_diff += [diff]
        a_chisq += [chisq]
        print(f'{r:04d}: - diff {diff:.3e}\tchisq {chisq:.3e}',end='\r')
        if max_iter is not None:    #: limit in iterations 
            if r > max_iter: 
                print(f'Routine stops due to the limit in iterations: {r} reached')
                break
        stop_cond = r<max_iter if Dn == 0 else diff >= Dn*thr_mul#/I.sum()
    print(f'\nTime: {time.time()-start_time} s')
    print()
    plt.figure(figsize=(10,14))
    plt.title('Convergence of R.-L. algorithm',fontsize=20)
    plt.plot(a_diff,'.-')
    plt.xlabel('$r$',fontsize=18)
    plt.ylabel('$\\Delta I_{(r)}^{(r+1)}$',fontsize=18)
    # plt.axhline(Dn,0,1,color='orange')
    # plt.ylim(1e-7,2e-6)
    # plt.axhline(Dn/sigma.mean(),0,1,color='violet')
    # plt.axhline(np.mean(a_diff),0,1)
    plt.figure(figsize=(10,14))
    plt.title('Chisq')
    plt.plot(a_chisq,'.-')
    # plt.axhline(np.mean(a_chisq),0,1)
    plt.show()
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
    rec_field = Sr(Sr1,Ir1).copy()[pad_size:-pad_size,pad_size:-pad_size]

    ## Plotting
    if display_fig:
        fast_image(rec_field,'Restored Field',**kwargs)
        fast_image(rec_field, norm='log',**kwargs)
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.set_title('Science Frame',fontsize=20)
        field_image(fig,ax1,field)
        ax2.set_title('Restored Field',fontsize=20)
        field_image(fig,ax2,rec_field)
        plt.show()
    return rec_field

def light_recover(dec_field: NDArray, thr: float, mean_bkg: float, ker_sigma: tuple[float,float], sub_frame: tuple[slice,slice] = (slice(None),slice(None)),binning: int = 63, results: dict | None = None, **search_args) -> tuple[ndarray,ndarray]:
    default_param = {
        'max_size': 5, 
        'cntrl': None, 
        'cntrl_sel': 'bright', 
        'debug_plots': False,
        'log': 'True'
        }
    for key, val in default_param.items():
        if key not in search_args.keys():
            search_args[key] = val
    tmp_field = dec_field[sub_frame]
    objs, errs, pos = searching(tmp_field,thr,mean_bkg,**search_args)
    maxvals = np.array([o.max() for o in objs])# - mean_bkg
    maxerrs = np.array([e[peak_pos(o)] for o, e in zip(objs,errs)])
    sigma,Dsigma = ker_sigma
    rec_brt  = maxvals*np.sqrt(2*np.pi)*sigma
    Drec_brt = rec_brt * np.sqrt((maxerrs/maxvals)**2 + (Dsigma/sigma)**2 )
    sort_args = np.argsort(rec_brt)[::-1]
    _ = dist_corr(pos,binning=binning,display_plots=True)
    if results is not None:
        results['objs'] = objs
        results['errs'] = errs
        results['pos']  = pos
    return rec_brt[sort_args], Drec_brt[sort_args]

def cutting(obj: NDArray, centre: Sequence[int], err: NDArray | None = None, debug_plots: bool = False) -> tuple[NDArray, NDArray | None, NDArray] | tuple[None, None, None]:
    """To select only pixels whithin the HWHM

    Parameters
    ----------
    obj : NDArray
        selected object
    centre : Sequence[int]
        coordinates of the centre
    debug_plots : bool, default False
        parameter to check the goodness of the algorithm

    Returns
    -------
    cut_obj : NDArray | None
        reshaped object
        Function returns `None` values whether the selected pixels
        are "few", that is the size (along either x or y direction)
        is less than 2 pixels
    shift : NDArray | None
        change in pixel coordinates 
        Given (x,y) the coordinates of 1 pixel in the
        `obj` matrix, then (x0,y0) the ones in the `cut_obj` 
        matrix are: 
        ```
            x0 = x - shift[0]
            y0 = y - shift[1]
        ```
        Function returns `None` values whether the selected pixels
        are "few", that is the size (along either x or y direction)
        is less than 2 pixels
    """
    if err is not None:
        if err.shape != obj.shape:
            print(err.shape,obj.shape)
            raise Exception('Vergognaci')    
    WRONG_RESULT = None, None, None   #: the returned value for rejected objects
    xdim, ydim = obj.shape      #: sizes of the object matrix
    x0, y0 = centre             #: centre coordinates
    val0 = obj[x0,y0]
    hm = val0/2          #: half maximum

    # find the value which best approximate the hm
    hm_pos = max(minimum_pos(np.abs(np.where(obj != val0,obj,0)-hm)))
    # compute the best approximation of the HWHM
    hwhm = np.rint((abs(hm_pos-x0) + abs(hm_pos-y0))/2).astype(int)
    if hwhm < 1:   #: check the size
        print('! HWHM nO')
        print(hwhm)
        #?
        if debug_plots:        fast_image(obj,'! HWHM small')
        #?
        return WRONG_RESULT
    xends = (max(x0-hwhm,0), min(x0+hwhm +1 , xdim))
    yends = (max(y0-hwhm,0), min(y0+hwhm +1 , ydim))
    # select only the pixels inside hwhm
    # cut = lambda centre, dim : slice(max(centre-hwhm,0), min(centre+hwhm +1 , dim))     #: function to compute the slices
    cut_obj = obj[slice(*xends),slice(*yends)].copy()
    if 1 in cut_obj.shape or 2 in cut_obj.shape:    #: check size
        #?
        if debug_plots: fast_image(cut_obj,'! Sizes')
        #?
        print('! No shape')
        return WRONG_RESULT
    print('\n\thwhm',hwhm)
    print('\thm_pos', hm_pos)
    print('\tsigma',hwhm / (2*np.log(2)))
    print('\tdim : ', xdim, ydim)
    print('\tcen : ', x0, y0)
    print('\tx : ', *xends)
    print('\ty : ', *yends)
    print('\tval0',val0)
    # compute the shift to trasform the coordinates
    shift = np.array([xends[0], yends[0]])
    print('\tcen : ', x0 - shift[0], y0 - shift[1])
    print('\tshift',shift)
    print('\tval1',cut_obj[x0-shift[0],y0-shift[1]])
    cut_err = err[slice(*xends),slice(*yends)].copy() if err is not None else None
    return cut_obj, cut_err, shift

def average_trend(obj: NDArray, centre: tuple[int, int], debgug_plots: bool = False) -> tuple[NDArray, NDArray]:
    """To compute the averaged trend of the light value from the centre pixel

    The function averages over the values of pixels at the same 
    distance from the centre

    Parameters
    ----------
    obj : NDArray
        selected object
    centre : tuple[int, int]
        coordinates of the centre

    Returns
    -------
    px : NDArray
        distances from the centre 
    mean_obj : NDArray
        mean trend of the light value from the centre
    
    See also
    --------
    stuff.distance
    """
    xdim, ydim = obj.shape      #: object shapes
    # define a method to compute the distance from the centre
    centre_dist = lambda pixel : distance(pixel, centre)
    # compute distances
    dist = np.array([centre_dist((i,j)) for i,j in zip(*np.meshgrid(np.arange(ydim), np.arange(xdim))[::-1])])
    # remove duplicates
    px = np.unique(dist)
    # average over values at the same distance
    mean_obj = np.array([ np.mean(obj[dist == d]) for d in px])

    if debgug_plots:
        n_dist = np.array([centre_dist((i,j)) for i,j in zip(*np.meshgrid(np.arange(ydim), np.arange(xdim))[::-1])])
        # remove duplicates
        n_px = np.unique(n_dist)
        # average over values at the same distance
        n_mean_obj = np.array([ np.mean(obj[n_dist == d]) for d in n_px])
        plt.figure()
        plt.title('Transverse')
        plt.plot(n_px,n_mean_obj,'.--')
    return px, mean_obj

def object_check(obj: NDArray, index: tuple[int,int], thr: float, sigma: Sequence[int] | None, err: NDArray | None = None, mode: Literal["bright", "low"] = 'bright', maxpos: tuple[int,int] | None = None, debug_plots: bool = False,**kwargs) -> tuple[NDArray, NDArray | None, tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] | None:
    """To check whether an object is acceptable as a star or not

    Parameters
    ----------
    obj : NDArray
        selected object
    index : tuple[int,int]
        coordinates in the frame of the board
    thr : float
        star object must have values over a certain threshold
    sigma : Sequence[int] | None
        collection of estimated STDs
    mode : Literal[&quot;bright&quot;, &quot;low&quot;], optional
        _description_, by default 'bright'
    maxpos : tuple[int,int] | None, optional
        _description_, by default None
    debug_plots : bool, optional
        _description_, by default False

    Returns
    -------
    tuple[NDArray, tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] | None
        _description_
    """
    if 'log' not in kwargs.keys():
        kwargs['log'] = False
    SIZE = 5    #:
    init_obj = np.copy(obj)
    c_obj = np.copy(obj)
    xmax, ymax = peak_pos(c_obj)      #: max value coordinates
    xdim, ydim = c_obj.shape          #: sizes
    #?
    if maxpos is not None:          #: check  
        xxmax, yymax = maxpos
        if xxmax != xmax or yymax != ymax:
            if kwargs['log']:
                print('real:', xxmax, yymax)
                print('used:', xmax, ymax)
            #?
            if debug_plots:
                fig, ax = plt.subplots(1,1)
                field_image(fig,ax,c_obj)
                ax.plot(ymax,xmax,'.b')
                ax.plot(yymax,xxmax,'.r')
                plt.show()
            #?
            raise
    #?
    val0 = c_obj[xmax,ymax]
    hm = val0/2          #: half maximum

    # find the value which best approximate the hm
    hm_pos = max(minimum_pos(np.abs(np.where(c_obj != val0,c_obj,0)-hm)))
    # compute the best approximation of the HWHM
    hwhm = np.rint((abs(hm_pos-xmax) + abs(hm_pos-ymax))/2).astype(int)
    if hwhm <= 1: mode = 'low'
    

    ### Bright object 
    if mode == 'bright':
        ## Cut 
        if kwargs['log']: print('\n\tFirst cut')
        # set the centre
        centre = np.array([xmax, ymax])
        # select the pixels of interest from `obj`
        cut_obj, cut_err, shift = cutting(c_obj, centre, err=err, debug_plots=debug_plots)
        # reject null object
        if cut_obj is None: return None
        if err is not None:
            if cut_err.shape != cut_obj.shape: 
                print(cut_err.shape,cut_obj.shape)
                raise IndexError('OH!')
        if cut_err is not None: shape0 = cut_err.shape
        # fit with a gaussian in order to find an approximation for the centroid
        pop, _ = new_kernel_fit(cut_obj-thr, err=cut_err, display_fig=False)
        if cut_err is not None:
            if cut_err.shape != shape0:
                raise ValueError('EH NO')
        if pop is None: return None
        # check sigma
        est_sigma = pop[1]
        if est_sigma <= 0:  #: negative sigma is unacceptable
            if kwargs['log']: print('BAD SIGMA')
            #?
            if debug_plots: fast_image(cut_obj,'Bad Sigma')
            #?
            return None
        coord = np.rint(pop[-2:]).astype(int)
        if debug_plots:
            fig,ax = plt.subplots(1,1)
            field_image(fig,ax,cut_obj)
            ax.plot(*coord[::-1],'xb')
            plt.show()
        # check the centroid position
        if 0 <= coord[0] < cut_obj.shape[0] and 0 <= coord[1] < cut_obj.shape[1]:
            if kwargs['log']: print('\n\tSecond cut')
            # compute centre coordinates in the frame of `obj`
            centre = coord + shift
            # select the pixels of interest from `obj`
            cut_obj, cut_err, shift = cutting(c_obj, centre, err=err, debug_plots=debug_plots)
            if cut_obj is None:     #: check
                return None
            if err is not None:
                if cut_err.shape != cut_obj.shape: 
                    print(cut_err.shape,cut_obj.shape)
                    raise IndexError('OH!')
        # change coordinates reference
        x0, y0 = centre - shift
        if debug_plots:
            fig,ax = plt.subplots(1,1)
            field_image(fig,ax,cut_obj)
            ax.plot(y0,x0,'xb')
            ax.plot(*peak_pos(cut_obj)[::-1],'xr')
            plt.show()
        # compute the value of the centroid
        val0 = cut_obj[x0, y0]
        xdim, ydim = cut_obj.shape          #: sizes of the selected portion
        hwhm = (max(abs(xdim-x0),x0) + max(abs(ydim-y0),y0))/2
        sigma += [hwhm]

        ## Gradient
        if kwargs['log']: print('cut_dim : ', xdim, ydim)
        # compute the mean trend
        px, mean_obj = average_trend(cut_obj, (x0,y0),debgug_plots=debug_plots)
        if debug_plots:
            plt.figure()
            plt.title('Trend')
            plt.plot(cut_obj[:,y0])
            plt.figure()
            plt.plot(cut_obj[x0,:])
            plt.show()
        # compute the first derivative
        grad1 = np.diff(mean_obj) 
        # check the derivative sign around the centre
        mean_width = np.rint(max(hwhm, 3)).astype(int)
        g_pos = np.where(grad1[:mean_width+1] >= 0)[0]
        #?
        if debug_plots:
            fig0, ax0 = plt.subplots(1,1)
            ax0.set_title('Mean obj')
            ax0.plot(px,mean_obj,'.--',color='blue')
            m_px = (px[:-1] + px[1:])/2
            ax0.plot(m_px, grad1, 'v--', color='orange')
            ax0.plot((m_px[:-1]+m_px[1:])/2, np.diff(grad1), '^--', color='green')
            ax0.axhline(0,0,1,color='black')
            ax0.axhline(np.mean(mean_obj[1:mean_width+1]),0,1,color='blue',linestyle='dashed')
            ax0.axhline(thr,0,1,color='red',linestyle='dotted')
            plt.show()
        #?
        if len(g_pos) == 0:
            # convert coordinates to the initial frame
            x0, y0 = np.array([x0, y0]) + shift
            # check the derivative sign out from the bulk
            g_pos = np.where(grad1[mean_width:] >= 0)[0]
            if len(g_pos) != 0:     #: cut the object
                xdim, ydim = c_obj.shape  #: initial sizes
                # cut at the first positive value of the derivative
                mean_width = g_pos[0]
                if kwargs['log']: print('Quite')
                # compute the size of the object from the centre
                xsize = ( max(0, x0-mean_width),  min(xdim, x0+mean_width+1))
                ysize = ( max(0, y0-mean_width),  min(xdim, y0+mean_width+1))
                # cut the object
                obj = c_obj[slice(*xsize), slice(*ysize)].copy()
                if err is not None:
                    err = err[slice(*xsize), slice(*ysize)]
                #?
                if debug_plots:                fast_image(obj,'cutted')
                #?
            if kwargs['log']: print('GOOD')
        else:
            ## S/N
            # compute the ratio between pixels around the centre and mean background
            ratio = np.mean(mean_obj[1:mean_width+1])/thr
            if kwargs['log']: print('S/N',ratio*100,'%')
            #?
            if debug_plots:            plt.show()
            #?
            if ratio >= 1:
                if kwargs['log']: print('Quite Quite Good')
                # convert coordinates to the initial frame
                x0, y0 = np.array([x0, y0]) + shift
                xdim, ydim = c_obj.shape  #: initial sizes
                if kwargs['log']: print('Quite')
                # compute the size of the object from the centre
                xsize = ( max(0, x0-mean_width),  min(xdim, x0+mean_width+1))
                ysize = ( max(0, y0-mean_width),  min(xdim, y0+mean_width+1))
                print('HEY',mean_width,xsize,ysize)
                # cut the object
                obj = c_obj[slice(*xsize), slice(*ysize)].copy()
                if err is not None:
                    err = err[slice(*xsize), slice(*ysize)]
                #?
                if debug_plots: fast_image(c_obj,'cutted S/N')
                #?
            else:
                if kwargs['log']:
                    print('RATIO',val0/thr*100,'%')
                    print('NO GOOD')
                return None
    elif mode == 'low':
        #?
        if debug_plots:        fast_image(obj,'LOW before cutting')
        #?
        if kwargs['log']: print('LOW')
        hwhm = np.rint(np.mean(sigma)).astype(int) if len(sigma) != 0 else SIZE
        cut = lambda centre, dim : slice(max(0, centre-hwhm), min(dim, centre + hwhm + 1))
        centre = np.array([xmax, ymax])
        x0, y0 = centre
        if kwargs['log']:
            print(max(0, x0-hwhm), min(xdim, x0 + hwhm + 1))
            print(max(0, y0-hwhm), min(ydim, y0 + hwhm + 1))
        cut_obj = c_obj[cut(x0, xdim), cut(y0, ydim)].copy()
        cut_err = err[cut(x0, xdim), cut(y0, ydim)].copy() if err is not None else None
        #?
        if debug_plots:        fast_image(cut_obj,'cut_obj')
        #?
        cxmax, cymax = peak_pos(cut_obj)
        shift = np.array([xmax - cxmax, ymax - cymax])
        if kwargs['log']:
            print('\n\thwhm', hwhm)
            print('\tsigma', hwhm / (2*np.log(2)))
            print('\tdim : ', xdim, ydim)
            print('\tcen : ', x0, y0)
            print('\tx : ', max(x0-hwhm,0), min(x0+hwhm+1, xdim))
            print('\ty : ', max(y0-hwhm,0), min(y0+hwhm+1, ydim))
        # fit with a gaussian in order to find the centroid
        pop, _ = new_kernel_fit(cut_obj-thr, err=cut_err, display_fig=False)
        if pop is None: return None
        # check sigma
        est_sigma = pop[1]
        if est_sigma <= 0:  #: negative sigma is unacceptable
            if kwargs['log']: print('BAD SIGMA')
            #?
            if debug_plots:            fast_image(cut_obj,'BAD SIGMA')
            #?
            return None
        coord = np.rint(pop[-2:]).astype(int)
        # check the centroid position
        if 0 <= coord[0] < cut_obj.shape[0] and 0 <= coord[1] < cut_obj.shape[1]:
            centre = coord + shift
            x0, y0 = centre
            if kwargs['log']:
                print(max(0, x0-hwhm), min(xdim, x0 + hwhm + 1))
                print(max(0, y0-hwhm), min(ydim, y0 + hwhm + 1))
            cut_obj = c_obj[cut(x0, xdim), cut(y0, ydim)].copy()
            cut_err = err[cut(x0, xdim), cut(y0, ydim)].copy() if err is not None else None
            #?
            if debug_plots:            fast_image(cut_obj,'Fit cut_obj')
            #?
            cxmax, cymax = peak_pos(cut_obj)
            shift = np.array([xmax - cxmax, ymax - cymax])
        #?
        if debug_plots:
            fig, ax = plt.subplots(1,1)
            field_image(fig, ax, cut_obj)
            plt.show()
        #?
        if kwargs['log']:
            print('zero',x0,y0)
            print('shift', shift)
        # change coordinates reference
        x0, y0 = centre - shift
        if kwargs['log']:
            print('zero',x0,y0)
            print('shape',cut_obj.shape)
        val0 = cut_obj[x0,y0]
        xdim, ydim = cut_obj.shape
        px, mean_obj = average_trend(cut_obj, (x0,y0),debgug_plots=debug_plots)
        if debug_plots:
            plt.figure()
            plt.title('Trend')
            plt.plot(cut_obj[:,y0])
            plt.figure()
            plt.title('Trend')
            plt.plot(cut_obj[:,x0])
            plt.show()
        grad1 = np.diff(mean_obj)
        #?
        if debug_plots:
            fig0,ax0 = plt.subplots(1,1)
            ax0.set_title('Low: mean obj')
            ax0.plot(px,mean_obj,'.--',color='blue')
            m_px = (px[:-1] + px[1:])/2
            ax0.plot(m_px, grad1, 'v--', color='orange')
            ax0.plot((m_px[:-1]+m_px[1:])/2, np.diff(grad1), '^--', color='green')
            ax0.axhline(0,0,1,color='black')
            ax0.axhline(np.mean(mean_obj[1:]),0,1,color='blue',linestyle='dashed')
            ax0.axhline(thr,0,1,color='red',linestyle='dotted')
            plt.show()
        #?
        ratio = np.mean(mean_obj[1:])/thr
        if ratio >= 1:
            g_pos = np.where(grad1 >= 0)[0]
            if np.mean(grad1) < 0:#len(g_pos) == 0:
                obj = cut_obj.copy()
                err = cut_err.copy() if cut_err is not None else None
            else:
                if kwargs['log']:
                    print('gradient')
                    print('NO GOOD')
                return None                
        else:
            if kwargs['log']:
                print(f'RATIO {val0/thr:%} %')
                print('NO GOOD')
            return None
    if 0 in obj.shape:
        print('> STOP NO SHAPE')
        fast_image(init_obj)
        return -1
        print('position',index)
        raise Exception(f'Zero shape ')
    # compute the coordinates of the centre and the edges on the board
    #. the changing of coordinates is obtained from the
    #. maximum coordinates
    xdim, ydim = obj.shape
    x, y = index
    xmax, ymax = peak_pos(obj)
    if debug_plots:
        fig,ax = plt.subplots(1,1)
        field_image(fig,ax,obj)
        ax.plot(y0,x0,'xb')
        ax.plot(ymax,xmax,'xr')
        plt.show()
        
    index = (x0 + (x-xmax), y0 + (y-ymax))
    xsize = (x-xmax, xdim + (x-xmax))
    ysize = (y-ymax, ydim + (y-ymax))
    if err is not None:
        if err.shape != obj.shape:
            print(err.shape,obj.shape)
            raise IndexError('NOOOOO')
    return obj, err, index, (xsize, ysize)
                
def art_obj(prb_obj: NDArray, index: tuple[int,int], bkg_val: float, errs: NDArray | None = None, ker_sigma: float | None = None, debug_plots: bool = False) -> tuple[NDArray,NDArray] | tuple[None,None]:
    fit_obj = prb_obj.copy() - bkg_val
    # prb_x, prb_y = index
    avg_cen = index 
    def gauss_func(pos, *args):
        xpos, ypos = pos
        k, s, x0, y0 = args
        zx = (xpos-x0)/s
        zy = (ypos-y0)/s
        return k * np.exp(-zx**2/2) * np.exp(-zy**2/2)
    xdim, ydim = prb_obj.shape
    # xrange, yrange = np.meshgrid(np.arange(ydim),np.arange(xdim))
    yrange, xrange = np.meshgrid(np.arange(ydim),np.arange(xdim))
    from scipy.optimize import curve_fit
    try:
        k0 = fit_obj[index]
    except:
        if debug_plots:
            fast_image(fit_obj)
        print(index)
        raise
    if ker_sigma is None:
        hm = k0/2
        hm_xpos, hm_ypos = minimum_pos(abs(hm-fit_obj))
        hwhm = np.sqrt((hm_xpos - index[0])**2 + (hm_ypos - index[1])**2)
        ker_sigma = hwhm   
    initial_values = [k0,ker_sigma,index[0],index[1]]  
    xfit = np.vstack((xrange.ravel(),yrange.ravel()))
    yfit = fit_obj.ravel()
    sigma = errs.ravel().copy() if errs is not None else None

    try:
        pop, pcov = curve_fit(gauss_func,xfit,yfit,initial_values,sigma=sigma)
        print(pop,np.sqrt(pcov.diagonal()))
    except RuntimeError:
        try:
            if debug_plots:        
                plt.figure()
                plt.title('runtime0')
                plt.imshow(fit_obj)
                plt.plot(index[1],index[0],'.')
                plt.show()
            print('Another Chance')
            _ = new_kernel_fit(fit_obj,display_fig=debug_plots)
            exit()
            # pop, pcov = curve_fit(gauss_func,xfit,yfit,initial_values,sigma=None)
            # print(pop,np.sqrt(pcov.diagonal()))
        except RuntimeError:
            if debug_plots:        
                plt.figure()
                plt.title('runtime')
                plt.imshow(fit_obj)
                plt.show()
            print('RuntimeError in rec_obj')
            return None, None
    except ValueError:
        print('YOHEY',errs.shape,fit_obj.shape)
        raise
    if pop[1] < 1: return None,None
    rec_obj = gauss_func((xrange,yrange),*pop) + bkg_val
    print('VARIANCE',np.sqrt(np.var(rec_obj-prb_obj)))
    rec_err = np.full(rec_obj.shape,np.sqrt(np.var(rec_obj-prb_obj)))
    if rec_obj.shape != prb_obj.shape:
        print('Probe',xdim,ydim)
        print(xrange)
        print(yrange)
        print(avg_cen)
        print(rec_obj.shape)
        fast_image(prb_obj)
        fast_image(rec_obj)
    print('CENVAL',prb_obj[avg_cen])
    print('CENVAL',rec_obj[avg_cen])
    dim = 2*int(4*pop[1])+1
    cen = dim // 2
    yrange, xrange = np.meshgrid(np.arange(dim),np.arange(dim))
    new_obj = gauss_func((xrange,yrange),pop[0],pop[1],cen,cen)
    new_err = np.full(new_obj.shape,np.std(rec_obj-prb_obj))
    return new_obj, new_err

            
def searching(field: NDArray, thr: float, bkg_val: float, errs: NDArray | None = None, max_size: int = 5, min_dist: int = 0, ker_sigma: float | None = None, num_objs: int | None = None, cntrl_mode: Literal['bright', 'low', 'all'] = 'all', debug_plots: bool = False, cntrl: int | None = None, cntrl_sel: str | None = None, display_fig: bool = False, **kwargs) -> None | tuple[list[NDArray], list[NDArray] | None, NDArray]:
    if 'log' not in kwargs.keys():
        kwargs['log'] = False
    if 'debug_check' not in kwargs.keys():
        kwargs['debug_check'] = False
    debug_check = kwargs['debug_check']
    kwargs.pop('debug_check')
    def info_print(cnt: int, index: tuple, peak: float) -> None:
        x0 , y0 = index
        if kwargs['log']:
            print(f'\n- - - -\nStep {cnt}')
            print(f'\tcoor : ({x0}, {y0})')
            print(f'\tpeak : {peak}')
    tmp_field = field.copy()
    display_field = field.copy()
    sigma = []
    arr_pos = np.empty((2,0),dtype=int)
    acc_obj = []
    err_obj = []
    acc_pos = np.empty((2,0),dtype=int)
    rej_obj = []
    rej_pos = np.empty((2,0),dtype=int)
    
    # first step
    xmax, ymax = peak_pos(tmp_field)    #: coordinates of the maximum
    peak = tmp_field[xmax, ymax]        #: maximum value
    stop_val = thr                      #: threashold
    cnt = 1
    obj_cnt = 0
    print('\n- - - SEARCHING START - - -')
    print(f'Stop_val : {stop_val}')
    info_print(cnt,(xmax, ymax), peak)
    while peak > stop_val:
        # debug_plots = True if cnt == 2 else False
        # debug_check = True if cnt == 2 else False
        rec_obj = None
        # compute an estimation of the size of the object
        xsize, ysize = new_grad_check(tmp_field, (xmax, ymax), thr, size=max_size)
        # compute slices
        x = slice(xmax - xsize[0], xmax + xsize[1]+1)
        y = slice(ymax - ysize[0], ymax + ysize[1]+1)
        # define the object
        obj = field[x,y].copy()
        # if cnt == 2: 
        #     cxmax, cymax = peak_pos(obj)
        #     plt.figure()
        #     plt.imshow(obj)
        #     plt.plot(cymax,cxmax,'.')
        #     plt.plot(ysize[0],xsize[0],'.')
        err = errs[x,y].copy() if errs is not None else None
        # if obj_cnt == 0:
        #     print(xsize,ysize)
        #     fast_image(obj)
        if 0 in obj.shape: 
            fast_image(obj)
            fig, ax = plt.subplots(1,1)
            field_image(fig,ax,field)
            ax.plot(ymax,xmax,'.')
            plt.show()

        #?
        if debug_plots: fast_image(obj,'Object')
        #?
        if kwargs['log']: print('SHAPE: ',obj.shape)
        # remove small object
        if obj.shape[0] <= 3 or obj.shape[1] <= 3:
                if kwargs['log']: 
                    print('xsize',xsize)
                    print('ysize',ysize)
                    print('diff',ymax-ysize)
                    print('Shape no Good')
                x0, y0 = xmax, ymax
                rej_obj += [obj]
                rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                #? 
                if debug_plots:          
                    fig0, ax0 = plt.subplots(1,1)
                    ax0.set_title('Rejected')
                    field_image(fig0,ax0,display_field)
                    if len(acc_pos[0]) != 0:
                        ax0.plot(acc_pos[1],acc_pos[0],'.b')
                    ax0.plot(rej_pos[1],rej_pos[0],'.r')
                    ax0.plot(y0,x0,'xr')
                    plt.show()
                #?
        else:
            if cntrl_sel is not None and cntrl_sel == 'low': debug_plots = False
            cxmax, cymax = peak_pos(obj)
            # if cnt == 3: 
            #     plt.figure()
            #     plt.imshow(obj)
            #     plt.plot(cymax,cxmax,'.')
            #     plt.plot(ysize[0],xsize[0],'.')
            #     plt.show()
            if kwargs['log']:
                print('          c s')
                print('x compare',cxmax,xsize)
                print('y compare',cymax,ysize)
            remove_cond = False
            while cxmax != xsize[0] or cymax != ysize[0]:
                #?
                if debug_plots:    
                    ff, aa = plt.subplots(1,1)
                    ff.suptitle('Have to reduce')            
                    field_image(ff,aa,obj)
                    aa.plot(cymax,cxmax,'.r')
                    aa.plot(ysize[0],xsize[0],'.b')
                    plt.show()
                #?
                row = min(cxmax, obj.shape[0]-cxmax-1)
                col = min(cymax, obj.shape[1]-cymax-1)
                condition = row <= col if all([cxmax != xsize[0], cymax != ysize[0]]) else cymax == ysize[0]
                if condition:
                    if kwargs['log']: print('row',cxmax, obj.shape[0]-cxmax-1)
                    xsize = np.array([xsize[0]-cxmax-1, xsize[1]]) if cxmax < xsize[0] else np.array([xsize[0], cxmax-xsize[0]-1])
                    if kwargs['log']: print(xsize)
                else:
                    if kwargs['log']: print('col',cymax, obj.shape[1]-cymax-1)
                    ysize = np.array([ysize[0]-cymax-1, ysize[1]]) if cymax < ysize[0] else np.array([ysize[0], cymax-ysize[0]-1])
                    if kwargs['log']: print(ysize)
                if kwargs['log']:
                    print('x compare',cxmax,xsize)
                    print('y compare',cymax,ysize)
                if (xsize < 0).any() or (ysize < 0).any():
                    raise Exception('Void object')
                # compute slices
                x = slice(xmax - xsize[0], xmax + xsize[1]+1)
                y = slice(ymax - ysize[0], ymax + ysize[1]+1)
                # define the object
                obj = field[x,y].copy()
                if errs is not None:
                    err = errs[x,y].copy()
                if (obj.shape[0] <= 3) or (obj.shape[1] <= 3):
                    if kwargs['log']: print('remove')
                    remove_cond = True
                    break
                cxmax, cymax = peak_pos(obj)
            if remove_cond:            
                if kwargs['log']: print('New Shape is too small')
                x0, y0 = xmax, ymax
                rej_obj += [obj]
                rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                #?
                if debug_plots:
                    fig0, ax0 = plt.subplots(1,1)
                    ax0.set_title('Rejected')
                    field_image(fig0,ax0,display_field)
                    if len(acc_pos[0]) != 0:
                        ax0.plot(acc_pos[1],acc_pos[0],'.b')
                    ax0.plot(rej_pos[1],rej_pos[0],'.r')
                    ax0.plot(y0,x0,'xr')
                    plt.show()
                #?
            elif peak/2 >= bkg_val:       #: bright objects
                #?
                if debug_plots:                
                    fast_image(obj,'Object before check')
                #?
                if kwargs['log']: print(f'\tshape : {obj.shape}')     
                #?
                if debug_plots:       
                    fig0, ax0 = plt.subplots(1,1)
                    field_image(fig0,ax0,display_field)
                    ax0.plot(ymax,xmax,'.')
                #?
                # check if object is acceptable
                check = object_check(obj, (xmax, ymax), bkg_val, sigma, err=err, debug_plots=debug_plots,**kwargs)
                if check is None or check == -1:
                    print('Check',check)
                    if kwargs['log']: print('Check is not good 1')
                    x0, y0 = xmax, ymax
                    rej_obj += [obj]
                    rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                    #?
                    if debug_plots:
                        fig0, ax0 = plt.subplots(1,1)
                        ax0.set_title('Rejected')
                        field_image(fig0,ax0,display_field)
                        if len(acc_pos[0]) != 0:
                            ax0.plot(acc_pos[1],acc_pos[0],'.b')
                        ax0.plot(rej_pos[1],rej_pos[0],'.r')
                        ax0.plot(y0,x0,'xr')
                        plt.show()
                    #?
                # elif check == -1:
                #     break
                else:
                    obj, err, (x0, y0), (xsize, ysize) = check
                    # if cnt == 2:
                    #     print('YO',xsize,ysize)
                    #     plt.figure()
                    #     plt.imshow(obj)
                    #     plt.plot(y0-ysize[0],x0-xsize[0],'.')
                    #     plt.show()
                       
                    if kwargs['log']:
                        print('xsize',xsize)
                        print('ysize',ysize)
                    # compute slices
                    x = slice(*xsize)
                    y = slice(*ysize)
                    #?
                    if debug_plots:
                        figg, axx = plt.subplots(1,2)
                        axx[0].set_title('Before selection')
                        field_image(figg,axx[0],obj)
                        field_image(figg,axx[1],display_field[x,y])
                        plt.show()
                    #?
                    # check 
                    if selection(obj,(x0, y0), arr_pos, max_size,  mindist=min_dist, sel='all',debug_check=debug_check):
                        xcen = x0 - x.indices(xmax)[0]
                        ycen = y0 - y.indices(ymax)[0]
                        rec_obj, rec_err = art_obj(obj,(xcen,ycen),bkg_val=bkg_val,errs=err,ker_sigma=ker_sigma,debug_plots=debug_plots)
                        # acc_obj += [obj]
                        if rec_obj is not None:
                            obj_cnt += 1
                            acc_obj += [rec_obj]
                            err_obj += [rec_err]
                            acc_pos = np.append(acc_pos, [[x0], [y0]], axis=1)
                            cen = rec_obj.shape[0] // 2
                            xends = (max(0,x0-cen), min(len(tmp_field),x0+cen+1))
                            yends = (max(0,y0-cen), min(len(tmp_field),y0+cen+1))
                            x = slice(*xends)
                            y = slice(*yends)
                            r_xends = (cen-min(x0,cen), min(len(tmp_field)-x0,cen+1)+cen)
                            r_yends = (cen-min(y0,cen), min(len(tmp_field)-y0,cen+1)+cen)
                            r_x = slice(*r_xends)
                            r_y = slice(*r_yends)
                            if np.diff(xends) != np.diff(r_xends) or np.diff(yends) != np.diff(r_yends):
                                print('OOOh')
                                print(xends,yends)
                                print(r_xends,r_yends)
                                raise
                            try:
                                display_field[x,y] -= rec_obj[r_x,r_y]
                            except ValueError:
                                print('OOOh')
                                print(xends,yends)
                                print(r_xends,r_yends)                                
                                plt.figure()
                                plt.imshow(display_field[x,y])
                                plt.figure()
                                plt.imshow(rec_obj[x,y])
                        else:
                            rej_obj += [obj]
                            rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)

                        #?
                        if debug_plots:
                            fig0, ax0 = plt.subplots(1,1)
                            ax0.set_title('Accepted')
                            field_image(fig0,ax0,display_field)
                            if len(rej_pos[0]) != 0:
                                ax0.plot(rej_pos[1],rej_pos[0],'.r')
                            ax0.plot(acc_pos[1],acc_pos[0],'.b')
                            ax0.plot(y0,x0,'xb')
                            plt.show()
                        #?
                    else:
                        if kwargs['log']: print('No for selection')
                        rej_obj += [obj]
                        rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                        #?
                        if debug_plots:
                            fig0, ax0 = plt.subplots(1,1)
                            ax0.set_title('Rejected')
                            field_image(fig0,ax0,display_field)
                            if len(acc_pos[0]) != 0:
                                ax0.plot(acc_pos[1],acc_pos[0],'.b')
                            ax0.plot(rej_pos[1],rej_pos[0],'.r')
                            ax0.plot(y0,x0,'xr')
                            plt.show()
                        #?
            else:       #: faint objects
                
                if cntrl_sel is not None and cntrl_sel == 'low': debug_plots = True
                #?
                if debug_plots:
                    fig0, ax0 = plt.subplots(1,1)
                    field_image(fig0,ax0,display_field)
                    ax0.plot(ymax,xmax,'.')
                #?
                # check whether the object is acceptable
                check = object_check(obj, (xmax, ymax), bkg_val, sigma, mode='low', err=err, maxpos=(xsize[0],ysize[0]), debug_plots=debug_plots)
                if check is None:
                    if kwargs['log']: print('Check is not good 2')
                    x0, y0 = xmax, ymax
                    rej_obj += [obj]
                    rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                    #?
                    if debug_plots:                    
                        fig0, ax0 = plt.subplots(1,1)
                        ax0.set_title('Rejected')
                        field_image(fig0,ax0,display_field)
                        if len(acc_pos[0]) != 0:
                            ax0.plot(acc_pos[1],acc_pos[0],'.b')
                        ax0.plot(rej_pos[1],rej_pos[0],'.r')
                        ax0.plot(y0,x0,'xr')
                        plt.show()
                    #?
                elif check == -1:
                    x0, y0 = xmax, ymax
                    rej_obj += [obj]
                    rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                    print('No for -1')
                    #?
                    if debug_plots:                    
                        fig0, ax0 = plt.subplots(1,1)
                        ax0.set_title('Rejected')
                        field_image(fig0,ax0,display_field)
                        if len(acc_pos[0]) != 0:
                            ax0.plot(acc_pos[1],acc_pos[0],'.b')
                        ax0.plot(rej_pos[1],rej_pos[0],'.r')
                        ax0.plot(y0,x0,'xr')
                        plt.show()
                    #?
                    break
                else:
                    obj, err, (x0, y0), (xsize, ysize) = check
                    if kwargs['log']:
                        print('xsize',xsize)
                        print('ysize',ysize)
                    # compute slices
                    x = slice(*xsize)
                    y = slice(*ysize)
                    #?
                    if debug_plots:
                        figg, axx = plt.subplots(1,2)
                        field_image(figg,axx[0],obj)
                        field_image(figg,axx[1],display_field[x,y])
                        plt.show()
                    #?
                    if selection(obj,(x0, y0), arr_pos, max_size,  mindist=min_dist, sel='all',debug_check=debug_check):
                        xcen = x0 - x.indices(xmax)[0]
                        ycen = y0 - y.indices(ymax)[0]
                        rec_obj, rec_err = art_obj(obj,(xcen,ycen),bkg_val=bkg_val,errs=err,ker_sigma=ker_sigma, debug_plots=debug_plots)
                        if rec_obj is not None:
                            obj_cnt += 1
                            acc_obj += [rec_obj]
                            err_obj += [rec_err]
                            acc_pos = np.append(acc_pos, [[x0], [y0]], axis=1)
                            cen = rec_obj.shape[0] // 2
                            xends = (max(0,x0-cen), min(len(tmp_field),x0+cen+1))
                            yends = (max(0,y0-cen), min(len(tmp_field),y0+cen+1))
                            x = slice(*xends)
                            y = slice(*yends)
                            r_xends = (cen-min(x0,cen), min(len(tmp_field)-x0,cen+1)+cen)
                            r_yends = (cen-min(y0,cen), min(len(tmp_field)-y0,cen+1)+cen)
                            r_x = slice(*r_xends)
                            r_y = slice(*r_yends)
                            if np.diff(xends) != np.diff(r_xends) or np.diff(yends) != np.diff(r_yends):
                                print('OOOh')
                                print(xends,yends)
                                print(r_xends,r_yends)
                                raise
                            try:
                                display_field[x,y] -= rec_obj[r_x,r_y]
                            except ValueError:
                                print('OOOh')
                                print(xends,yends)
                                print(r_xends,r_yends)                                
                                plt.figure()
                                plt.imshow(display_field[x,y])
                                plt.figure()
                                plt.imshow(rec_obj[x,y])
                        else:
                            if kwargs['log']: print('rec_obj is None')
                            rej_obj += [obj]
                            rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                        #?
                        if debug_plots:
                            fig0, ax0 = plt.subplots(1,1)
                            ax0.set_title('Accepted')
                            field_image(fig0,ax0,display_field)
                            if len(rej_pos[0]) != 0:
                                ax0.plot(rej_pos[1],rej_pos[0],'.r')
                            ax0.plot(acc_pos[1],acc_pos[0],'.b')
                            ax0.plot(y0,x0,'xb')
                            plt.show()
                        #?
                    else:
                        if kwargs['log']: print('No good for selection low obj')
                        rej_obj += [obj]
                        rej_pos = np.append(rej_pos, [[x0], [y0]], axis=1)
                        #?
                        if debug_plots:
                            fig0, ax0 = plt.subplots(1,1)
                            ax0.set_title('Rejected')
                            field_image(fig0,ax0,display_field)
                            if len(acc_pos[0]) != 0:
                                ax0.plot(acc_pos[1],acc_pos[0],'.b')
                            ax0.plot(rej_pos[1],rej_pos[0],'.r')
                            ax0.plot(y0,x0,'xr')
                            plt.show()
                        #?
        # collect coordinates of the centre
        arr_pos = np.append(arr_pos, [[x0], [y0]], axis=1)
        
        if rec_obj is None:
            tmp_field[x,y] = 0.0
        else:
            try:
                # update the field
                tmp_field[x,y] -= rec_obj[r_x,r_y]
            except:
                print(x,y)
                print(x0,y0)
                print(xcen,ycen)
                fast_image(rec_obj)
                fast_image(obj)
                fast_image(tmp_field[x,y])
                print(tmp_field[x,y].shape)
                print(obj.shape)
                print(rec_obj.shape)
                raise

        old_data = (xmax, ymax)
        # compute the next step
        xmax, ymax = peak_pos(tmp_field)
        #!
        if old_data == (xmax,ymax): 
            edge = lambda centre : slice(max(0,centre-4),min(len(field),centre+4))
            f, a = plt.subplots(1,2)
            f.suptitle('Ripetition')
            field_image(f,a[0],display_field)
            field_image(f,a[1],display_field[edge(arr_pos[0,-1]),edge(arr_pos[1,-1])])
            a[0].plot(arr_pos[1,-1],arr_pos[0,-1],'.')
            fast_image(rej_obj[-1],'Bad')
            raise Exception('! RIPETITION !')
        #!
        peak = tmp_field[xmax, ymax]   
        cnt += 1
        info_print(cnt,(xmax, ymax), peak)
        #?
        if debug_plots: fast_image(tmp_field,'tmp_field')
        #?
        if cnt == cntrl:
            print('Stop for control')
            break
        if obj_cnt == num_objs:
            break
                
    fig, ax = plt.subplots(1,1)
    ax.set_title('Frame after objects extraction',fontsize=20)
    field_image(fig,ax,display_field)
    if len(acc_pos[0]) != 0:
        ax.plot(acc_pos[1],acc_pos[0],'.b')
    if len(rej_pos[0]) != 0:
        ax.plot(rej_pos[1],rej_pos[0],'.r')
    if 0 not in acc_pos:
        ax.plot(acc_pos[1],acc_pos[0],'.b')
    if 0 not in rej_pos:
        ax.plot(rej_pos[1],rej_pos[0],'.r')
    plt.show()
    fig, ax = plt.subplots(1,1)
    ax.set_title('Searching algorithm',fontsize=20)
    field_image(fig,ax,field)
    if 0 not in acc_pos:
        ax.plot(acc_pos[1],acc_pos[0],'.b')
    plt.show()
    if 0 in acc_pos.shape: return None
    return acc_obj, err_obj, acc_pos
