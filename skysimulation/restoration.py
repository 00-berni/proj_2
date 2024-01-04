import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray,ArrayLike
from scipy.signal import find_peaks
from typing import Callable, Any
from .display import fast_image, field_image
from .field import Gaussian, N, Uniform, noise
from .field import K as field_const

def peak_pos(field: NDArray) -> int | tuple[int,int]:
    """Finding the coordinate/s of the maximum

    :param field: array
    :type field: NDArray

    :return: coordinate/s
    :rtype: int | tuple[int,int]
    """
    if len(field.shape) == 1:
        return field.argmax()
    else:
        return np.unravel_index(field.argmax(),field.shape)

def fit_routine(xdata: NDArray, ydata: NDArray, method: Callable[[NDArray,Any],NDArray], initial_values: list[float] | NDArray, err: NDArray | None = None, sel: str = 'pop', print_res: bool = True, names: list[str | None] = [] ,**kwargs) -> list[NDArray | float]:
    from scipy.optimize import curve_fit
    pop, pcov = curve_fit(method, xdata, ydata, initial_values,sigma=err,**kwargs)
    Dpop = np.sqrt(pcov.diagonal())
    if err is not None:
        fit = method(xdata,*pop)
        chisq = (((ydata-fit)/err)**2).sum()
        chi0 = len(ydata) - len(pop)
        # Dchi0 = np.sqrt(2*chi0)
    if print_res:
        if len(names) == 0:
            names = [f'pop{i+1}' for i in range(len(pop))]
        print('- FIT RESULTS -')
        for i in range(len(pop)):
            print('\t'+names[i]+f' = {pop[i]} +- {Dpop[i]}')
        if err is not None:
            print(f'\tred_chi = {chisq/chi0*100} +- {np.sqrt(2/chi0)*100} %')
        print('- - - -')
    results = []
    if sel == 'all' or 'pop' in sel:
        results += [pop, Dpop]
    if sel == 'all' or 'pcov' in sel:
        results += [pcov]
    if sel == 'all' or 'chisq' in sel:
        results += [chisq] 
    return results


##*
def dark_elaboration(params: tuple[str, float | tuple], iteration: int = 3, dim: int = N, display_fig: bool = False, **kwargs) -> NDArray:
    """The function computes a number (`iteration`) of darks
    and averages them in order to get a mean estimation 
    of the detector noise

    :param params: parameters of the signal
    :type params: tuple[str, float  |  tuple]
    :param iteration: number of darks, defaults to 3
    :type iteration: int, optional
    :param dim: size of the field, defaults to N
    :type dim: int, optional
    :param display_fig: if `True` pictures are shown, defaults to False
    :type display_fig: bool, optional
    
    :return: mean dark
    :rtype: NDArray
    """
    # generating the first dark
    dark = noise(params, dim=dim)
    # making the loop
    for _ in range(iteration-1):
        dark += noise(params, dim=dim)
    # averaging
    dark /= iteration
    if display_fig:
        if 'title' not in kwargs:
            kwargs['title'] = f'Dark elaboration\nAveraged on {iteration} iterations'
        fast_image(dark,**kwargs)
    return dark

def bkg_est(field: NDArray, display_fig: bool = False) -> float:
    """Estimating a value for the background

    :param field: the field
    :type field: NDArray
    :param display_fig: if `True` pictures are shown, defaults to False
    :type display_fig: bool, optional
    
    :return: estimated background value
    :rtype: float
    """
    from .field import K
    # flattening the field
    field = np.copy(field).flatten() 
    # checking the field
    field = field[np.where(field > 0)[0]]
    num = np.sqrt(len(field))
    print(field.max())
    # binning
    # bins = np.linspace(field.min(),field.max(),len(field)//3)
    counts, bins = np.histogram(field,bins=num.astype(int))
    # normalization coefficient
    k = (counts.sum()*np.diff(bins).mean())
    # position of the maximum
    maxpos = counts.argmax()
    #? checking 
    if maxpos == 0: maxpos = counts[1:].argmax()
    # maximum value
    maxval = bins[maxpos]    
    # taking the bin next the maximum
    shift = 1
    while bins[maxpos+shift] == maxval:
        shift += 1
    # storing this value
    ebkg = bins[maxpos+shift]

    ## Old sequence
    tmp = counts[maxpos+1:]
    dist = abs(tmp[:-2] - tmp[2:])
    pos = np.where(counts == tmp[dist.argmax()+2])[0]
    mbin = (max(bins[pos])+bins[maxpos])/2
    ##

    ## Gaussian fit
    # saving the mean
    mean = (maxval+ebkg)/2
    # hm = counts.max()//2
    # cutbin = bins[:maxpos]
    # cutcnt = counts[:maxpos]
    # try:
    #     hm = abs(cutcnt-hm).argmin()
    # except ValueError:
    #     print('Error')
    #     print(maxpos)
    #     print(counts.argmax())
    #     print(len(cutcnt))
    #     raise
    # hw = mean - cutbin[hm]
    # print(hw/K)
    # sigma = hw / np.sqrt(2*np.log(2))
    # print('sigma ',sigma)
    # print('sigma ',sigma/K)

    # def gauss_fit(x,*args):
    #     sigma, mu = args
    #     return np.exp(-((x-mu)/sigma)**2/2)
    
    # from scipy.optimize import curve_fit
    # from scipy.stats import norm
    # # k = counts.max() 
    # initial_values = [sigma,mean]
    # pop, pcov = curve_fit(gauss_fit,cutbin,cutcnt,initial_values)
    # s, m = pop
    # Ds, Dm = np.sqrt(pcov.diagonal())
    # print('curve fit')
    # print('mu ',m/K,Dm/K)
    # print('sigma ',s/K,Ds/K)
    # # print('k ',k,Dk)
    # data = np.sort(field)
    # mid = abs(data-mean).argmin()
    # data = data[:2*mid]
    # (mu, sigma) = norm.fit(data,loc=mean,scale=sigma,method='MM')
    # print('Different fit')
    # print('mean ',mu/K)#,Dmu/K)
    # print('sigma ',sigma/K)#,Dsigma/K)
    ##

    if display_fig:
        # bins = np.log10(bins)
        plt.figure(figsize=(14,10))
        plt.stairs(counts, bins,fill=True)
        # plt.axvline(np.log10(ebkg),0,1,linestyle='--',color='red')
        plt.axvline(mean,0,1,linestyle='--',color='red')
        if shift != 1:
            plt.axvline(maxval,0,1,linestyle='--',color='yellow')
        # plt.axvline(max(bins[pos]),0,1,linestyle='--',color='yellow')
        # plt.axvline((bins[counts.argmax()]+field.min())/2,0,1,linestyle='--',color='blue')
        # plt.axvline(mbin,0,1,linestyle='--',color='orange')
        # plt.axvline(cutbin[hm],0,1,color='green',alpha=0.5)
        # plt.axvline(2*mean - cutbin[hm],0,1,color='green',alpha=0.5)
        # plt.axhline(cutcnt[hm],0,1,color='black',alpha=0.5)
        # xx = np.linspace(cutbin.min()/2,2*mu-cutbin.min()/2,500)
        # yy = gauss_fit(xx,sigma,mu)
        # from scipy.integrate import quad
        # gauss = lambda x : gauss_fit(x,sigma,mu)
        # plt.plot(xx,yy/quad(gauss,xx.min(),xx.max())[0]*k,'black',linewidth=2)
        plt.xlabel('$F_{sn}$')
        plt.ylabel('counts')
        # plt.xscale('log')
        plt.show()
    return mean

def moving(direction: str, field: NDArray, index: tuple[int,int], back: float, size: int = 3, acc: float = 1e-5) -> list[int]:
    """Looking in one direction

    `direction` is a string and contains two parameters:

      1. `'f'` or `'b'` mean forward or backward, respectively.
      2. `'x'` or `'y'` mean horizontal or vertical direction, respectively.

    It is also possible to combine different directions such as `'fxby'`
    (combinations along same axis are not allowed) 

    :param direction: selected direction
    :type direction: str
    :param field: the field
    :type field: NDArray
    :param index: the coordinates of the object
    :type index: tuple[int,int]
    :param size: maximum size of the object, defaults to 3
    :type size: int, optional
    
    :return: the size of the object in different directions
    :rtype: list[int]
    """
    print(f':: Results of moving() func for {direction} direction ::')
    tmp_field = field.copy()
    dim = len(tmp_field)
    # pixel coordinates
    x, y = index
    # size += 1
    
    # list to store results
    results = []
    # inizializing the variable for the direction
    #    1 : forward
    #    0 : ignored
    #   -1 : backward
    xd = 0
    yd = 0
    # initializing the limits
    xmax = x
    ymax = y
    # initializing the conditions on x and y 
    xcond = lambda xval, xlim: True
    ycond = lambda yval, ylim: True
    
    # forward along x
    if 'fx' in direction:
        print('hey')
        # computing the edge
        xmax = min(size, dim-1-x)
        # impossible movement
        if xmax == 0: 
            results += [0]
        # updating the condition on x
        else: 
            xd = 1
            xcond = lambda xval, xlim: xval < xlim
    # backward along x
    elif 'bx' in direction:
        # computing the edge
        xmax = min(size, x)
        # impossible movement
        if xmax == 0: results += [0]
        # updating the condition on x
        else: 
            xd = -1
            xcond = lambda xval, xlim: xval < xlim 
    
    # forward along y
    if 'fy' in direction:
        # computing the edge
        ymax = min(size, dim-1-y)
        # impossible movement
        if ymax == 0: results += [0]
        # updating the condition on y
        else: 
            yd = 1
            ycond = lambda yval, ylim: yval < ylim 
    # backward along y
    elif 'by' in direction:
        # computing the edge
        ymax = min(size, y)
        # impossible movement
        if ymax == 0: results += [0]
        # updating the condition on y
        else: 
            yd = -1
            ycond = lambda yval, ylim: yval < ylim 
    
    print('1 result',results)
    # if there are no forbidden directions
    if xd != 0 or yd != 0:
        # inizilizing the variables for the size
        xsize = 0
        ysize = 0
        condition = xcond(xsize,xmax) and ycond(ysize,ymax)
        print(xcond(xsize,xmax),xmax)
        # routine to compute the size
        while condition:
            # near pixels
            step0 = tmp_field[x + xsize*xd, y + ysize*yd]
            step1 = tmp_field[x + (xsize+1)*xd, y + (ysize+1)*yd]
            # gradient
            grad = step1 - step0
            # ratio
            ratio = step1/step0
            # condition to stop
            if step1 == 0 or step1 < back or (ratio - 1) >= acc:
                print('ratio',ratio)
                print('grad and step',grad,step1)
                print('size',xsize,ysize)
                break
            # condition to go on
            else:
                xsize += 1
                ysize += 1
                condition = xcond(xsize,xmax) and ycond(ysize,ymax)
        # saving the results
        if 'x' in direction and xd != 0: results = [xsize] + results
        if 'y' in direction and yd != 0: results += [ysize]
    
    if len(results) == 1: results = results[0] 
    print('2 result',results)
    print(':: End ::')
    return results


def grad_check(field: NDArray, index: tuple[int,int], back: float, size: int = 3) -> tuple[NDArray,NDArray]:
    """Checking the gradient trend around an object

    :param field: the field
    :type field: NDArray
    :param index: coordinates of the object
    :type index: tuple[int,int]
    :param size: the maximum size of the object, defaults to 3
    :type size: int, optional
    
    :return: size of the object (x,y)
    :rtype: tuple[NDArray,NDArray]
    """
    mov = lambda val: moving(val,field,index,back,size)
    xy_dir = ['fxfy','fxby','bxfy','bxby'] 
    a_xysize = np.array([mov(dir) for dir in xy_dir])
    print(':: Resutls of grad_check() ::')
    print('sizes',a_xysize)
    xf_size = a_xysize[:2,0]
    xb_size = a_xysize[2:,0]
    yf_size = a_xysize[(0,2),1]
    yb_size = a_xysize[(1,3),1]

    a_size = np.array([[[mov('fx'),*xf_size],[mov('bx'),*xb_size]],
                       [[mov('fy'),*yf_size],[mov('by'),*yb_size]]])
    print('matrix',a_size)
    x_size, y_size = a_size.max(axis=2)
    print(':: End ::')
    return x_size, y_size
    
def selection(obj: NDArray, index: tuple[int,int], apos: NDArray, size: int, mindist: int = 5, minsize: int = 3, cutsize: int = 5) -> bool:
    """Selecting the objects for the fit

    :param objs: list of extracted objects
    :type objs: list[NDArray]
    :param apos: positions array
    :type apos: NDArray
    :param size: max selected size
    :type size: int
    :param maxdist: max accepted distance between objects, defaults to 5
    :type maxdist: int, optional
    :return: list of the selected and rejected objects 
    :rtype: tuple[list[NDArray], list[None | NDArray]]
    """
    obj = np.copy(obj)
    dim = len(obj)
    if size != 1 and dim <= minsize: 
        print(f'\t:Selection:\tdim = {dim}')
        return False

    if len(apos[0]) > 0:    
        #: method to compute the length
        dist = lambda x,y: np.sqrt(x**2 + y**2)
        # extracting positions
        x, y = index
        xi, yi = np.copy(apos[:,:])
        # computing distances
        mindist += size
        adist = np.array( [dist(xi[i]-x, yi[i]-y) for i in range(len(xi))] )
        pos = np.where(np.logical_and(adist <= mindist, adist != 0))  
        if len(pos[0]) != 0: 
            print(f'\t:Selection:\tdist = {adist[pos]} - adist = {adist} - mindist = {mindist}')
            return False
        del x, y, xi, yi, dist, adist, pos
    
    x, y = peak_pos(obj)
    if abs(x - dim//2) > 1 or abs(y - dim//2) > 1: 
        print(f'\t:Selection:\t(x,y) = ({x},{y}) - dim//2 = {dim//2}')
        return False 
    lim = cutsize if dim > 2*cutsize else dim//2 
    for i in range(lim):
        i += 1
        r, l, u, d = np.copy(obj[[x+i,x-i,x,x],[y,y,y+1,y-1]])
        diff1 = abs(r/l - u/d) 
        diff2 = abs(r/d - l/u)
        if diff1 >= 0.3 or diff2 >= 0.3 or (diff1 > 0.25 and diff2 > 0.25): 
            print(f'\t:Selection:\tdiff1 = {diff1} -  diff2 = {diff2}')
            return False

    return True


##*
def object_isolation(field: NDArray, thr: float, size: int = 3, objnum: int = 10, reshape: bool = False, reshape_corr: bool = False, sel_cond: bool = False, display_fig: bool = False,**kwargs) -> NDArray | None:
    """To isolate the most luminous star object.
    The function calls the `size_est()` function to compute the size of the object and
    then to extract it from the field.

    :param field: the field
    :type field: NDArray
    :param thr: threshold value
    :type thr: float
    :param size: maximum size of the object, defaults to 3
    :type size: int, optional
    :param objnum: maximum number of object to search, defaults to 10
    :type objnum: int, optional
    :param reshape: if `True` x and y sizes of the object are equal, defaults to False
    :type reshape: bool, optional
    :param reshape_corr: if `True` objects at the edges are corrected, defaults to False
    :type reshape_corr: bool, optional
    :param display_fig: if `True` pictures are shown, defaults to False
    :type display_fig: bool, optional
    
    :return: the extracted objects or `None`
    :rtype: NDArray | None
    """
    tmp_field = field.copy()
    display_field = field.copy()

    a_pos = np.array([[],[]],dtype=int)
    extraction = []
    sel_pos = np.array([[],[]],dtype=int)
    rej_obj = []
    rej_pos = np.array([[],[]],dtype=int)
    
    if display_fig: 
        tmp_kwargs = {key: kwargs[key] for key in kwargs.keys() - {'title'}} 

    k = 0 
    ctrl_cnt = 0
    while k < objnum:
        # finding the peak
        index = peak_pos(tmp_field)
        ctrl = False
        if 0 in index: 
            print(index)
            ctrl = True
        peak = tmp_field[index]
        # checking the value
        if peak <= thr:
            break
        # computing size
        x, y = index
        a_size = grad_check(tmp_field,index,thr,size)
        if ctrl: 
            print(index)
            print(a_size)
            # raise Exception('Ci fermiamo un attimo') 
        print(f':: Iteration {k} of object_isolation :: ')
        print('a_size',a_size)
        x_size, y_size = a_size
        xu, xd = x_size
        yu, yd = y_size
        xr = slice(x-xd, x+xu+1) 
        yr = slice(y-yd, y+yu+1)
        print('Slices: ',xr,yr)
        tmp_field[xr,yr] = 0.0
        print('a_size 2',a_size)
        a_pos = np.append(a_pos,[[x],[y]],axis=1)
        if all([0,0] == a_size[0]) or all([0,0] == a_size[1]):
            rej_obj += [field[xr,yr].copy()]
            rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
            print(f'Rejected obj: ({x},{y})')
        else: 
            if reshape:
                a_size = np.array(a_size)
                pos = np.where(a_size != 0)
                print('POS: ',pos)
                min_size = a_size[pos].min()
                a_size[pos] = min_size
                x_size, y_size = a_size
                xu, xd = x_size
                yu, yd = y_size
                xr = slice(x-xd, x+xu+1) 
                yr = slice(y-yd, y+yu+1)
            
            obj = field[xr,yr].copy() 

            if reshape_corr and (0 in [xu,xd,yu,yd]):
                xpadu = 0 if xd != 0 else xu
                xpadd = 0 if xu != 0 else xd
                ypadu = 0 if yd != 0 else yu
                ypadd = 0 if yu != 0 else yd

                xpad_pos = (xpadu,xpadd)
                ypad_pos = (ypadu,ypadd)
                print('Pad',xpad_pos,ypad_pos)
                obj = np.pad(obj,(xpad_pos,ypad_pos),'reflect')
            
            save_cond = selection(obj,index,a_pos,size) if sel_cond else True
            
            if save_cond:
                print(f'** OBJECT SELECTED ->\t{k}')
                display_field[xr,yr] = 0.0
                
                extraction += [obj]
                sel_pos = np.append(sel_pos,[[x],[y]],axis=1)
                if display_fig:
                    tmp_kwargs['title'] = f'N. {k+1} object {index} - {ctrl_cnt}'
                k += 1 
            else: 
                print(f'!! OBJECT REJECTED ->\t{k}')
                rej_obj += [obj]
                rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
                if display_fig:
                    tmp_kwargs['title'] = f'Rejected object {index} - {ctrl_cnt}'

            if display_fig: 
                fast_image(obj,**tmp_kwargs) 
            
            if (ctrl_cnt >= objnum and len(extraction) >= 3) or ctrl_cnt > 2*objnum and sel_cond:
                break

            ctrl_cnt += 1

    if 'title' not in kwargs:
        kwargs['title'] = 'Field after extraction'
    fast_image(display_field,**kwargs)
    fast_image(tmp_field,**kwargs)    

    if sel_cond and len(extraction) > 0:
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

    if len(extraction) == 0: extraction = None
    print(':: End ::')
    return extraction

def kernel_fit(obj: NDArray, err: float | None = None, display_fig: bool = False, **kwargs) -> float:
    """Estimating the sigma of the kernel

    :param obj: extracted objects
    :type obj: NDArray
    :param back: estimated value of the background
    :type back: float
    :param noise: mean noise value
    :type noise: float
    
    :return: mean sigma of the kernel
    :rtype: float
    """
    dim = len(obj)
    m = np.arange(dim)
    x, y = np.meshgrid(m,m)
    xmax, ymax = peak_pos(obj)
    r = np.sqrt(x**2 + y**2)
    sigma0 = 1
    print('sigma',sigma0)
    k0 = obj.max()

    def fit_func(pos: NDArray,*args) -> NDArray:
        x, y = pos
        k, sigma = args
        x0, y0 = xmax, ymax 
        kernel = Gaussian(sigma)
        return k * kernel.value(x-x0)*kernel.value(y-y0)
    if err is not None:
        err = np.full(obj.shape,err,dtype=float)
    initial_values = [k0,sigma0]
    xfit = np.vstack((x.ravel(),y.ravel()))
    yfit = obj.ravel()
    errfit = err.ravel() if err is not None else err
    pop, Dpop = fit_routine(xfit[::-1],yfit,fit_func,initial_values,err=errfit,names=['k','sigma'])
    k, sigma = pop
    Dk, Dsigma = Dpop

    if display_fig:
        figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else None
        title = kwargs['title'] if 'title' in kwargs.keys() else ''
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.set_title(title)
        field_image(fig,ax,obj)
        mm = np.linspace(m[0],m[-1],100)
        xx, yy = np.meshgrid(mm,mm)
        ax.contour(xx,yy,fit_func((xx,yy),*pop),colors='b',linestyles='dashed',alpha=0.7)
        # kwargs.pop('title')
        # kwargs.pop('figsize')

        plt.figure()
        plt.subplot(1,2,1)
        plt.errorbar(m,obj[xmax,:],yerr=err[xmax,:],fmt='.')
        plt.plot(mm,fit_func([xmax,mm],*pop))
        plt.subplot(1,2,2)
        plt.errorbar(m,obj[:,ymax],yerr=err[:,ymax],fmt='.')
        plt.plot(mm,fit_func([mm,ymax],*pop))
        plt.show()
    return sigma

def kernel_estimation(extraction: list[NDArray], err: float, dim: int, selected: slice = slice(None), all_results: bool = False, display_plot: bool = False, **kwargs) -> NDArray | tuple[NDArray,tuple[float,float]]:
    """Estimation of the kernel from a Gaussian model

    :param extraction: extracted objects
    :type extraction: list[NDArray]
    :param back: estimated value of the background
    :type back: float
    :param noise: mean noise
    :type noise: float
    :param dim: size of the kernel
    :type dim: int
    :param display_plot: if `True` pictures are shown, defaults to False
    :type display_plot: bool, optional
    :param all_results: if `True` additional results are returned, defaults to False
    :type all_results: bool, optional
    
    :return: kernel (and sigma with the error)
    :rtype: NDArray | tuple[NDArray,tuple[float,float]]
    """
    sel_extr = [*extraction[selected]]
    a_sigma = np.array([],dtype=float)
    for obj in sel_extr:
        sigma = kernel_fit(obj,err,display_plot,**kwargs)
        a_sigma = np.append(a_sigma,sigma)
        del sigma
    sigma = np.mean(a_sigma)
    Dsigma = np.sqrt(((sigma-a_sigma)**2).sum()/(len(a_sigma)*(len(a_sigma)-1)))
    print(f'\nsigma = {sigma:.5f} +- {Dsigma:.5f}')
    kernel = Gaussian(sigma)
    kernel = kernel.kernel(dim)
    if display_plot:
        if 'title' not in kwargs:
            kwargs['title'] = 'Estimated kernel'
        fast_image(kernel,**kwargs)
    
    if all_results:
        return kernel,(sigma,Dsigma)
    else:
        return kernel


def LR_deconvolution(field: NDArray, kernel: NDArray, back: float, noise: float, iter: int = 17) -> NDArray:
    """Richardson-Lucy deconvolution algorithm

    :param field: the field
    :type field: NDArray
    :param kernel: estimated kernel
    :type kernel: NDArray
    :param back: estimated value of the background
    :type back: float
    :param noise: mean noise
    :type noise: float
    :param iter: number of iterations, defaults to 17
    :type iter: int, optional
    
    :return: recostructed field
    :rtype: NDArray
    """
    n = max(back,noise)
    pos = np.where(field <= n)
    tmp_field = field[pos].copy()
    n = np.mean(tmp_field)
    Dn = np.sqrt(np.mean((tmp_field-n)**2))
    
    from scipy.integrate import trapz
    from scipy.ndimage import convolve
    from skimage.restoration import richardson_lucy
    I = np.copy(field)
    P = np.copy(kernel)
    Ir = lambda S: convolve(S,P)
    Sr = lambda S,Ir: S * convolve(I/Ir,P)

    r = 1
    Ir0 = Ir(I)
    Sr1 = Sr(I,Ir0)
    Ir1 = Ir(Sr1)
    diff = abs(trapz(trapz(Ir1-Ir0)))
    print('Dn', Dn)
    print(f'{r:02d}: - diff {diff}')
    while r < iter: #diff > Dn:
        r += 1
        Sr0 = Sr1
        Ir0 = Ir1
        Sr1 = Sr(Sr0,Ir0)
        Ir1 = Ir(Sr1)
        diff = abs(trapz(trapz(Ir1-Ir0)))
        print(f'{r:02d}: - diff {diff}')
    SrD = Sr(Sr1,Ir1)
    fast_image(SrD)    
    Sr1 = richardson_lucy(I,P,iter)
    fast_image(Sr1)   
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(SrD,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(Sr1,cmap='gray')
    plt.show()
    fast_image(Sr1-back-noise)    
    return Sr1
