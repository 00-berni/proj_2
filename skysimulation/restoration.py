from typing import Callable, Sequence, Any
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray,ArrayLike
from scipy.signal import find_peaks
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

def fit_routine(xdata: NDArray, ydata: NDArray, method: Callable[[NDArray,Any],NDArray], initial_values: list[float] | NDArray, err: NDArray | None = None, sel: str = 'pop', print_res: bool = True, names: list[str | None] = [] ,**kwargs) -> list[NDArray | float ] | dict:
    """Routine for a fit with `curve_fit`

    :param xdata: x values
    :type xdata: NDArray
    :param ydata: y values
    :type ydata: NDArray
    :param method: model
    :type method: Callable[[NDArray,Any],NDArray]
    :param initial_values: guesses for parameters
    :type initial_values: list[float] | NDArray
    :param err: error values, defaults to `None`
    :type err: NDArray | None, optional
    :param sel: selected result/s, defaults to 'pop'
    :type sel: str, optional
    :param print_res: if `True` fit results are printed, defaults to `True`
    :type print_res: bool, optional
    :param names: names of the parameters, defaults to []
    :type names: list[str  |  None], optional
    
    :return: estimated parameters and additional info
    :rtype: list[NDArray | float]
    """
    # importing the function
    from scipy.optimize import curve_fit
    # computing the fit
    pop, pcov = curve_fit(method, xdata, ydata, initial_values,sigma=err,**kwargs)
    # extracting the errors
    Dpop = np.sqrt(pcov.diagonal())
    # computing the chi squared
    if err is not None:
        # evaluating function in `xdata`
        fit = method(xdata,*pop)
        # computing chi squared
        chisq = (((ydata-fit)/err)**2).sum()
        chi0 = len(ydata) - len(pop)
        # Dchi0 = np.sqrt(2*chi0)
    # print results
    if print_res:
        if len(names) == 0:
            names = [f'pop{i+1}' for i in range(len(pop))]
        print('- FIT RESULTS -')
        for i in range(len(pop)):
            print('\t'+names[i]+f' = {pop[i]} +- {Dpop[i]}\t{Dpop[i]/pop[i]*100:.2} %')
        if err is not None:
            print(f'\tred_chi = {chisq/chi0*100} +- {np.sqrt(2/chi0)*100} %')
        print('- - - -')
    # collecting results
    results = []
    if sel == 'all' or 'pop' in sel:
        results += [pop, Dpop]
    elif sel == 'dict' and len(names) != 0:
        names += ['D'+name for name in names]
        pop = np.append(pop,Dpop)
        dpop = {names[i]: pop[i] for i in range(len(names))}
        results = dpop            
    if sel == 'all' or 'pcov' in sel:
        results += [pcov]
    if sel == 'all' or 'chisq' in sel:
        results += [chisq] 
    # else: raise Exception('Wrong `sel`')
    return results

def mean_n_std(xdata: NDArray) -> tuple[float, float]:
    dim = len(xdata)
    mean = np.mean(xdata)
    std = np.sqrt(((xdata-mean)**2).sum()/(dim*(dim-1)))
    return mean, std

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

def moving(direction: str, field: NDArray, index: tuple[int,int], back: float, size: int = 3, acc: float = 1e-5) -> list[int] | int:
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
    x, y = index    #: pixel coordinates
    # size += 1
    
    results = []    #: list to store results
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
            if step0 == 0: 
                print(f'Error:\n\tx,y = {x,y}\n\txd,yd = {xd},{yd}\n\txsize,ysize = {xsize},{ysize}\nstep1 = {step1}')
                raise
            grad = step1 - step0    #: gradient
            ratio = step1/step0     #: ratio
            # condition to stop
            if step1 == 0 or step1 < back or (ratio - 1) > acc:
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

def grad_check(field: NDArray, index: tuple[int,int], back: float, size: int = 7, acc: float = 1e-5) -> NDArray:
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
    # defining method to move in a direction
    mov = lambda val: moving(val,field,index,back,size=size,acc=acc)
    # setting the diagonal directions
    xy_dir = ['fxfy','fxby','bxfy','bxby'] 
    a_xysize = np.array([mov(dir) for dir in xy_dir])
    print(':: Resutls of grad_check() ::')
    print('sizes',a_xysize)
    # extracting resukts
    xf_size = a_xysize[:2,0]
    xb_size = a_xysize[2:,0]
    yf_size = a_xysize[(0,2),1]
    yb_size = a_xysize[(1,3),1]
    # building a matrix of sizes
    a_size = np.array([[[mov('fx'),*xf_size],
                        [mov('bx'),*xb_size]],
                       [[mov('fy'),*yf_size],
                        [mov('by'),*yb_size]]])
    print('matrix',a_size)
    # taking the maxima
    a_size = a_size.max(axis=2)
    print(':: End ::')
    return a_size


def new_moving(direction: str, field: NDArray, index: tuple[int,int], back: float, size: int = 7) -> list[int] | int:
    dim = len(field)
    x,y = index
    results = []
    hm = field[index]/2
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
    
    # if there are no forbidden directions
    if xd != 0 or yd != 0:
        # inizilizing the variables for the size
        xsize = 1
        ysize = 1
        condition = xcond(xsize,xmax) and ycond(ysize,ymax)
        print(xcond(xsize,xmax),xmax)
        # routine to compute the size
        while condition:
            step = field[x+xd*xsize, y+yd*ysize]
            if step == 0: 
                if 'x' in direction and xd != 0: results  = [xsize+size] + results
                if 'y' in direction and yd != 0: results += [ysize+size]
                if len(results) == 1: results = results[0]
                return results
            elif step <= hm: 
                if 'x' in direction and xd != 0: results  = [xsize] + results
                if 'y' in direction and yd != 0: results += [ysize]
                if len(results) == 1: results = results[0]
                return results
            else:
                xsize += 1
                ysize += 1
                condition = xcond(xsize,xmax) and ycond(ysize,ymax)
        if 'x' in direction and xd != 0: results  = [-1] + results
        if 'y' in direction and yd != 0: results += [-1]

    if len(results) == 0: raise
    if len(results) == 1: results = results[0]
    return results

def new_grad_check(field: NDArray, index: tuple[int,int], back: float, size: int = 7, acc: float = 1e-5) -> NDArray:
    # 
    n_mov = lambda val : new_moving(val,field,index,back,size)
    xf_dir = ['fx','bx','fy','by']      #: directions
    a_xysize = np.array([n_mov(dir) for dir in xf_dir]) 
    print('A_XF',type(a_xysize))
    n_pos = np.where(a_xysize == -1)[0]
    if len(n_pos) != 0:
        dim = len(field)
        # a_xysize[n_pos] = np.array([mov(xf_dir[i]) for i in n_pos])
        a_xysize[n_pos] = np.array([ min(size, ((i+1)%2)*dim + (i%2*2-1)*(index[i//2])) for i in n_pos])
        print('A_XF',type(a_xysize),a_xysize)
    a_xysize = np.where(a_xysize > size, a_xysize-size, a_xysize)

    return a_xysize.reshape(2,2)


def selection(obj: NDArray, index: tuple[int,int], apos: NDArray, size: int, sel: str | Sequence[str] = 'all', mindist: int = 5, minsize: int = 3, cutsize: int = 5, acc: float = 1e-1, reshape: bool = False) -> bool:
    """Selecting the objects for the fit

    sel:
      * `'all'` : all
      * `'size'`: size
      * `'dist'`: distance
      * `'cut'` : cut

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
    # distance must be 
    if (size+mindist) < 0: raise
    cond = False        #: variable to check if `sel` is correct
    obj = np.copy(obj)
    xdim, ydim = obj.shape
    dim = min(xdim,ydim)
    if sel == 'all' or 'size' in sel:
        cond = True
        if size != 1 and dim <= minsize: 
            print(f'\t:Selection:\tdim = {dim}')
            return False
    if sel == 'all' or 'dist' in sel:
        cond = True
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
    if sel == 'all' or 'cut' in sel:    
        cond = True
        x, y = peak_pos(obj)
        if reshape:  
            if abs(x - xdim//2) > 1 or abs(y - ydim//2) > 1: 
                print(f'\t:Selection:\t(x,y) = ({x},{y}) - dim//2 = {dim//2}')
                return False 
        lims = np.array([x, xdim-1 - x, y, ydim-1 - y])
        print(obj.shape)
        print(x,y)
        print('LIMS',lims)
        ind = np.where(lims > 1)[0]
        if len(ind) == 0: 
            return False
        else:
            step0 = obj[x,y]
            step00 = obj[x,y]
            step0_0 = obj[x,y]
            lim = lims[ind].min()
            dirc = lambda i : i%2*2-1
            xpos = lambda x : dirc(x)*(1-x//2)
            ypos = lambda y : dirc(y)*(y//2)
            print('IND',ind)
            diag = np.array([[1,3],[0,2]])
            dpos = np.array([ j in ind for j in diag.flatten()]).reshape(2,2)
            ind0 = np.array([*ind])
            xind, yind = np.where(dpos == False)
            if len(xind) != 0:
                ind0 = np.delete(ind0, 2-xind-yind) if len(xind) == 1 else diag[(xind+1)%2]
            print('DPOS',dpos)
            print('IND0',ind0)
            diag = np.array([[0,3],[1,2]])
            ndpos = np.array([ j in ind for j in diag.flatten()]).reshape(2,2)
            ind1 = np.array([*ind])
            xind, yind = np.where(ndpos == False)
            if len(xind) != 0:
                ind1 = np.delete(ind1, (1-xind)*((yind+2)%3)+xind) if len(xind) == 1 else diag[(xind+1)%2]
            print('NDPOS',ndpos)
            print('IND1',ind1)
            del xind, yind
            for i in range(1,lim+1):
                values = np.array([obj[x+xpos(j)*i,y+ypos(j)*i] for j in ind])
                step1 = np.mean(values)
                ratio = step1/step0 - 1
                print('RATIO',ratio)
                if ratio > acc:
                    print('END RATIO')
                    return False
                step0 = step1

                if dpos.all(axis=1).any():
                    values = np.array([obj[x+dirc(j)*i,y+dirc(j)*i] for j in ind0])
                    step11 = np.mean(values) if len(values) > 1 else values[0]
                    ratio = step11/step00 - 1
                    print('RATIO 00',ratio)
                    if ratio > acc:
                        print('END RATIO')
                        return False
                    step00 = step11

                if ndpos.all(axis=1).any():
                    values = np.array([obj[x+dirc(j)*i,y-dirc(j)*i] for j in ind1])
                    step1_1 = np.mean(values) if len(values) > 1 else values[0]
                    ratio = step1_1/step0_0 - 1
                    print('RATIO 0_0',ratio)
                    if ratio > acc:
                        print('END RATIO')
                        return False
                    step0_0 = step1_1

        # ind = np.where(lims != 0)[0]
        # if len(ind) == 0 or len(ind) == 1: 
        #     return False
        # else:
            # lim = min(lims[ind].min(),2)
            # xpos = lambda x : (x%2*2-1)*(1-x//2)
            # ypos = lambda y : (y%2*2-1)*(y//2)
            # for i in range(1,lim+1):
            #     values = np.array([ obj[x + xpos(j)*i, y + ypos(j)*i] for j in ind])
            #     ratios = np.array([values[j]/values for j in range(1,len(values))])
            #     ratios = np.delete(ratios,np.where(ratios==1))
            #     rdim = len(ratios)
            #     if rdim == 0: break
            #     elif rdim == 1:
            #         diff = abs(ratios - 1)
            #     else:
            #         diff = np.array([abs(ratios[j] - ratios[(j+1)%rdim]) for j in range(rdim)])
            #     if np.count_nonzero(diff > acc) > len(diff)//2:
            #         print(f'{i} - DIFF = {diff}')
            #         return False
                # i += 1
                # r, l, u, d = np.copy(obj[[x+i,x-i,x,x],[y,y,y+1,y-1]])
                # #!! DA RIVEDERE
                # diff1 = abs(r/l - u/d) 
                # diff2 = abs(r/d - l/u)
                # if diff1 >= 0.3 or diff2 >= 0.3 or (diff1 > 0.25 and diff2 > 0.25): 
                #     print(f'\t:Selection:\tdiff1 = {diff1} -  diff2 = {diff2}')
                #     return False
    if cond: return True
    else: raise


##*
def object_isolation(field: NDArray, thr: float, size: int = 3, acc: float = 1e-1, objnum: int = 10, reshape: bool = False, reshape_corr: bool = False, sel_cond: bool = False, mindist: int = 5, minsize: int = 3, cutsize: int = 5, grad_new: bool = True, display_fig: bool = False,**kwargs) -> tuple[list[NDArray],NDArray] | None:
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
    :type objnum: int, optioFalsenal
    :param reshape: if `True` x and y sizes of the object are equal, defaults to False
    :type reshape: bool, optional
    :param reshape_corr: if `True` objects at the edges are corrected, defaults to False
    :type reshape_corr: bool, optional
    :param display_fig: if `True` pictures are shown, defaults to False
    :type display_fig: bool, optional
    
    :return: the extracted objects or `None`
    :rtype: NDArray | None
    """
    # copying the field
    tmp_field = field.copy()        #: field from which objects will be removed
    display_field = field.copy()    #: field from which only selected objects will be removed
    # initializing some useful variables
    a_pos = np.empty(shape=(2,0),dtype=int)     #: storing coordinates of studied objects
    extraction = []                             #: list to collect selected objects
    sel_pos = np.empty(shape=(2,0),dtype=int)   #: storing coordinates of selected objects
    rej_obj = []                                #: list to collect rejected objects
    rej_pos = np.empty(shape=(2,0),dtype=int)   #: storing coordinates of rejected objects
    # condition for the plots
    if display_fig: 
        tmp_kwargs = {key: kwargs[key] for key in kwargs.keys() - {'title'}} 
    # routine to extract objects
    k = 0           #: counter of selected objects
    ctrl_cnt = 0    #: counter of iterations
    while k < objnum:
        # finding the maximum value position
        index = peak_pos(tmp_field)
        
        ctrl = False    #?: solo per me
        #? per me
        if 0 in index: 
            print(index)
            ctrl = True
        #?
        
        peak = tmp_field[index]
        # condition to stop
        if peak <= thr:
            break
        # computing size
        a_size = new_grad_check(tmp_field,index,thr,size,acc) if grad_new else grad_check(tmp_field,index,thr,size,acc)
        
        #? per me
        if ctrl: 
            print(index)
            print(a_size)
        #?
        
        print(f':: Iteration {k} of object_isolation :: ')
        print('a_size',a_size)
        # removing the object from `tmp_field`
        x, y = index
        xu, xd, yu, yd = a_size.flatten()
        xr = slice(x-xd, x+xu+1) 
        yr = slice(y-yd, y+yu+1)
        print('Slices: ',xr,yr)
        tmp_field[xr,yr] = 0.0
        print('a_size 2',a_size)
        a_pos = np.append(a_pos,[[x],[y]],axis=1)
        # if the object is a single px it is rejected
        if any(([[0,0],[0,0]] == a_size).all(axis=1)):
            rej_obj += [field[xr,yr].copy()]
            rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
            print(f'Rejected obj: ({x},{y})')
        else: 
            if reshape:
                # reshaping the object in order to 
                # get the same size in each directions
                pos = np.where(a_size != 0)
                print('POS: ',pos)
                min_size = a_size[pos].min()
                a_size[pos] = min_size
                xu, xd, yu, yd = a_size.flatten()
                xr = slice(x-xd, x+xu+1) 
                yr = slice(y-yd, y+yu+1)
            # extracting the object
            obj = field[xr,yr].copy() 

            if reshape_corr and (0 in [xu,xd,yu,yd]):
                # reshaping the object in order to 
                # get a squared matrix
                xpadu = 0 if xd != 0 else xu
                xpadd = 0 if xu != 0 else xd
                ypadu = 0 if yd != 0 else yu
                ypadd = 0 if yu != 0 else yd

                xpad_pos = (xpadu,xpadd)
                ypad_pos = (ypadu,ypadd)
                print('Pad',xpad_pos,ypad_pos)
                # extending the object
                obj = np.pad(obj,(xpad_pos,ypad_pos),'reflect')
            
            # checking if the object is acceptable or not
            save_cond = selection(obj,index,a_pos,size,sel='all',mindist=mindist,minsize=minsize,cutsize=cutsize) if sel_cond else True
            if save_cond:
                print(f'** OBJECT SELECTED ->\t{k}')
                # updating the field to plot
                display_field[xr,yr] = 0.0
                # storing the selected object
                extraction += [obj]
                sel_pos = np.append(sel_pos,[[x],[y]],axis=1)
                if display_fig:
                    tmp_kwargs['title'] = f'N. {k+1} object {index} - {ctrl_cnt}'
                # updating the counter
                k += 1 
            else: 
                print(f'!! OBJECT REJECTED ->\t{k}')
                # storing the rejected object
                rej_obj += [obj]
                rej_pos = np.append(rej_pos,[[x],[y]],axis=1)
                if display_fig:
                    tmp_kwargs['title'] = f'Rejected object {index} - {ctrl_cnt}'
            # plotting the object
            if display_fig: 
                fig, ax = plt.subplots(1,1)
                field_image(fig,ax,tmp_field,**{key: tmp_kwargs[key] for key in tmp_kwargs.keys() - {'title'}})
                if len(rej_pos[0])!= 0:
                    ax.plot(rej_pos[1,:-1],rej_pos[0,:-1],'.r')
                    ax.plot(rej_pos[1,-1],rej_pos[0,-1],'x',color='red')
                if len(sel_pos[0])!=0:
                    ax.plot(sel_pos[1,:-1],sel_pos[0,:-1],'.b')
                    ax.plot(sel_pos[1,-1],sel_pos[0,-1],'xb')
                fast_image(obj,**tmp_kwargs) 
            # condition to prevent the stack-overflow
            if (ctrl_cnt >= objnum and len(extraction) >= 3) or ctrl_cnt > 2*objnum and sel_cond:
                break
            # updating the iteration counter
            ctrl_cnt += 1
    # plotting the field after extraction
    if 'title' not in kwargs:
        kwargs['title'] = 'Field after extraction'
    fast_image(display_field,**kwargs)
    fast_image(tmp_field,**kwargs)    
    # plotting the field with markers for rejected and selected objects
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
    # if nothing is taken or found
    if len(extraction) == 0: return None
    print(f'\nRej:\t{len(rej_obj)}\nExt:\t{len(extraction)}')
    print(':: End ::')
    return extraction, sel_pos

def err_estimation(field: NDArray, mean_val: float, thr: float | int = 30, display_plot: bool = False) -> tuple[float,float]:
    """Estimating the mean fluctuation of the background

    :param field: field matrix
    :type field: NDArray
    :param mean_val: mean value of background
    :type mean_val: float
    :param thr: threshold from which fit is computed in % of the maximum value, defaults to 30
    :type thr: float | int, optional
    :param display_plot: if `True` plots are shown, defaults to False
    :type display_plot: bool, optional
    
    :return: estimation of the mean fluctuation
    :rtype: float
    """
    # copying the field
    field = np.copy(field)
    # computing the ratio of each pixels with the others
    diff = np.array([i/field for i in field])
    print(f'diff = {diff.shape}')
    print(f'diff -> {diff[0]}')
    diff = diff.flatten()
    print(f'diff = {diff.shape}')
    #! da migliorare !
    # removing the ratio of same pixels
    pos = np.where(diff!=1.)[0]
    print(len(np.where(diff==1.)[0]))
    #!               !
    # computing the contrast
    diff = diff[pos] - 1
    print(f'diff = {diff.shape}')
    # computing the histogram
    bins = np.linspace(min(diff),max(diff),np.sqrt(len(diff)).astype(int)*2)
    counts, bins = np.histogram(diff,bins=bins)
    # computing the maximum
    maxpos = counts.argmax()
    maxval = (bins[maxpos+1] + bins[maxpos])/2

    def gauss_fit(x: float | NDArray,*args) -> float | NDArray:
        """Gaussian model for the fit

        :param x: variable
        :type x: float | NDArray

        :return: Gaussian value
        :rtype: float | NDArray
        """
        k,mu,sigma = args
        r = (x-mu)/sigma
        return k * np.exp(-r**2/2)

    # computing the threshold as the fraction of the maximum
    thr = counts[maxpos] * thr/100
    # taking only values over the threshold
    pos = np.sort(np.where(counts >= thr)[0])[[0,-1]]
    print('edges',pos)
    edges = bins[pos]
    print('edges',edges)
    print('maxval',maxval)
    cut = slice(pos[0],pos[1]+1)
    # computing the fit
    pop, Dpop = fit_routine(bins[cut],counts[cut],gauss_fit,[counts[maxpos],maxval,1],names=['k','mu','sigma'])
    pop[2] = abs(pop[2])
    mu = pop[1]
    print(f'Pop = {(mu+1)*np.mean(field)} - {mean_val}')
    sigma = pop[2]
    print(f'Sigma_pop = {sigma*mean_val} - {sigma*np.mean(field)} - {(sigma * np.sqrt(2*np.log(2))) * mean_val}')
    mean_val = np.mean(field)
    mean_val = (mu+1)*mean_val
    if display_plot:
        plt.figure()
        plt.title('Fluctuations')
        plt.stairs(counts,bins,fill=True, label='data')
        xx = np.linspace(min(bins),max(bins),1000)
        plt.plot(xx,gauss_fit(xx,*pop),color='green',label='fit')

        plt.axhline(thr,0,1,color='black',linestyle='dotted',label='threshold')
        plt.axvline(maxval,0,1,linestyle='--',color='red',label='max value')
        plt.axvline(mu,0,1,linestyle='--',color='violet',label='$\mu_{fit}$')
        plt.axvline(edges[0],0,1,color='pink')
        plt.axvline(edges[1],0,1,color='pink')
        
        plt.legend()
        plt.xlabel('$I_i/I_j$')
        plt.ylabel('counts')
        plt.show()
    # computing the mean fluctuation
    return (sigma * np.sqrt(2*np.log(2))) * mean_val, mean_val



def kernel_fit(obj: NDArray, err: float | None = None, size_cut: int | None = 9, display_fig: bool = False, **kwargs) -> tuple[NDArray,NDArray]:
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
    # data need to be flattened 
    if size_cut is not None:
        if len(obj) > size_cut:
            cut = (len(obj)-size_cut)//2
            obj = np.copy(obj[cut:-cut,cut:-cut])
            del cut
    dim = len(obj)
    m = np.arange(dim)
    x, y = np.meshgrid(m,m)
    xmax, ymax = peak_pos(obj)  #: maximum value coordinates

    def fit_func(pos: tuple[NDArray,NDArray],*args) -> NDArray:
        """Gaussian model for the fit

        :param pos: coordinates (x,y)
        :type pos: tuple[NDArray,NDArray]

        :return: value of the gaussian
        :rtype: NDArray
        """
        x, y = pos
        k, sigma = args
        x0, y0 = xmax, ymax
        kernel = Gaussian(sigma)
        return k * kernel.value(x-x0)*kernel.value(y-y0)
    
    # computing the error matrix
    if err is not None:
        err = np.full(obj.shape,err,dtype=float)
    # computing the fit
    sigma0 = 1
    print('sigma',sigma0)
    k0 = obj.max()
    initial_values = [k0,sigma0]
    xfit = np.vstack((x.ravel(),y.ravel()))
    yfit = obj.ravel()
    errfit = err.ravel() if err is not None else err
    pop, Dpop = fit_routine(xfit[::-1],yfit,fit_func,initial_values,err=errfit,names=['k','sigma'])    
    # extracting values
    k, sigma = pop
    Dk, Dsigma = Dpop
    print(f'\tDerivative: ')

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
        if err is not None:
            plt.figure()
            plt.subplot(1,2,1)
            plt.title('On x axis')
            plt.errorbar(m,obj[xmax,:],yerr=err[xmax,:],fmt='.')
            plt.plot(mm,fit_func([xmax,mm],*pop))
            plt.subplot(1,2,2)
            plt.title('On y axis')
            plt.errorbar(m,obj[:,ymax],yerr=err[:,ymax],fmt='.')
            plt.plot(mm,fit_func([mm,ymax],*pop))
        plt.show()
    return pop, Dpop


def new_kernel_fit(obj: NDArray, err: float | None = None, size_cut: int | None = 9, initial_values: list[int] | None = None, display_fig: bool = False, **kwargs) -> tuple[NDArray,NDArray]:
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

    xdim, ydim = obj.shape
    xmax, ymax = peak_pos(obj)  #: maximum value coordinates
    x = np.arange(xdim)
    y = np.arange(ydim) 
    y,x = np.meshgrid(y,x)

    def fit_func(pos: tuple[NDArray,NDArray],*args) -> NDArray:
        """Gaussian model for the fit

        :param pos: coordinates (x,y)
        :type pos: tuple[NDArray,NDArray]

        :return: value of the gaussian
        :rtype: NDArray
        """
        x, y = pos
        k, sigma = args
        x0, y0 = xmax, ymax
        kernel = Gaussian(sigma)
        return k * kernel.value(x-x0)*kernel.value(y-y0)
    
    # computing the error matrix
    if err is not None:
        err = np.full(obj.shape,err,dtype=float)
    # computing the fit
    if initial_values is None:
        sigma0 = 1
        print('sigma',sigma0)
        k0 = obj.max()
        initial_values = [k0,sigma0]
    xfit = np.vstack((x.ravel(),y.ravel()))
    yfit = obj.ravel()
    errfit = err.ravel() if err is not None else err
    pop, Dpop = fit_routine(xfit,yfit,fit_func,initial_values,err=errfit,names=['k','sigma'])    
    # extracting values
    k, sigma = pop
    Dk, Dsigma = Dpop

    if display_fig:
        figsize = kwargs['figsize'] if 'figsize' in kwargs.keys() else None
        title = kwargs['title'] if 'title' in kwargs.keys() else ''
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.set_title(title)
        field_image(fig,ax,obj)
        mx = np.linspace(0,xdim-1,50)
        my = np.linspace(0,ydim-1,50)
        yy, xx = np.meshgrid(my,mx)
        ax.contour(yy,xx,fit_func((xx,yy),*pop),colors='b',linestyles='dashed',alpha=0.7)
        ax.plot(ymax,xmax,'.r')
        # kwargs.pop('title')
        # kwargs.pop('figsize')
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

def kernel_estimation(extraction: list[NDArray], err: float, dim: int, selected: slice = slice(None), size_cut: int | None = 9, all_results: bool = False, display_plot: bool = False, **kwargs) -> NDArray | tuple[NDArray,tuple[float,float]]:
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
    # copying the list
    sel_extr = [*extraction[selected]]
    a_sigma = np.empty(shape=(0,2),dtype=float)  #: array to store values of sigma
    a_k = np.empty(shape=(0,2),dtype=float)      #: array to store values of k
    # computing an initial value for sigma and a threshold
    obj = sel_extr[0]
    xmax, ymax = peak_pos(obj)
    hm = obj[xmax,ymax]/2
    xmin, ymin = np.unravel_index(abs(obj-hm).argmin(),obj.shape)
    sigma0 = int(np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2))*2
    print(sigma0)
    del obj,xmax,ymax,xmin,ymin,hm
    # routine
    for obj in sel_extr:
        initial_values = [obj.max(),sigma0//4 if sigma0//4 != 0 else 1]
        pop, Dpop = new_kernel_fit(obj,err,size_cut=size_cut,initial_values=initial_values,display_fig=display_plot,**kwargs)
        k, sigma = pop
        Dk, Dsigma = Dpop
        if Dsigma/sigma < 2 and sigma <= sigma0:
            a_k = np.append(a_k,[[k,Dk]],axis=0)
            a_sigma = np.append(a_sigma,[[sigma,Dsigma]],axis=0)
        del sigma, Dsigma, k, Dk
    maxpos = a_sigma[:,0].argmax()
    maxval, maxerr = a_sigma[maxpos]
    discr = (a_sigma[:,0]-maxval)/maxerr
    errpos = np.where(discr > 1)[0]
    if len(errpos) != 0 and len(errpos) != len(a_sigma[:,0]):
        print(f'REMOVE - {errpos}')
        a_sigma = np.delete(a_sigma,errpos,axis=0)
    if len(a_sigma) == 0: raise
    elif len(a_sigma) == 1: sigma, Dsigma = a_sigma
    else:
        # computing the mean and STD
        sigma, Dsigma = mean_n_std(a_sigma[:,0])
    print(f'\nsigma = {sigma:.5f} +- {Dsigma:.5f} -> {Dsigma/sigma*100:.2} %')
    # computing the kernel
    kernel = Gaussian(sigma)
    kernel = kernel.kernel(dim)
    
    if 'title' not in kwargs:
        kwargs['title'] = f'Estimated kernel\n$\\sigma = $ {sigma}'
    fast_image(kernel,**kwargs)

    if all_results:
        return kernel, (a_sigma,a_k)
    else:
        return kernel, (sigma,Dsigma)


def LR_deconvolution(field: NDArray, kernel: NDArray, mean_val: float, iter: int = 10, sel: str = 'all', display_fig: bool = False, **kwargs) -> NDArray:
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
    pos = np.where(field <= mean_val)
    tmp_field = field[pos].copy()
    n = np.mean(tmp_field)
    Dn = np.sqrt(np.mean((tmp_field-n)**2))
    
    from scipy.integrate import trapezoid
    from scipy.ndimage import convolve
    from skimage.restoration import richardson_lucy
    I = np.copy(field)
    P = np.copy(kernel)
    if sel == 'all' or 'mine' in sel:
        Ir = lambda S: convolve(S,P)
        Sr = lambda S,Ir: S * convolve(I/Ir,P)

        r = 1
        Ir0 = Ir(I)
        Sr1 = Sr(I,Ir0)
        Ir1 = Ir(Sr1)
        diff = abs(trapezoid(trapezoid(Ir1-Ir0)))
        print('Dn', Dn)
        print(f'{r:02d}: - diff {diff}')
        while r < iter: #diff > Dn:
            r += 1
            Sr0 = Sr1
            Ir0 = Ir1
            Sr1 = Sr(Sr0,Ir0)
            Ir1 = Ir(Sr1)
            diff = abs(trapezoid(trapezoid(Ir1-Ir0)))
            print(f'{r:02d}: - diff {diff}')
        SrD = Sr(Sr1,Ir1)
        if display_fig: fast_image(SrD,**kwargs)    
        Sr = SrD
    if sel == 'all' or 'rl' in sel:
        Sr1 = richardson_lucy(I,P,iter)
        if display_fig: fast_image(Sr1,**kwargs)   
        Sr = Sr1
    if display_fig:
        if sel == 'all':
            fig, (ax1,ax2) = plt.subplots(1,2)
            field_image(fig,ax1,SrD,**kwargs)
            field_image(fig,ax2,Sr1,**kwargs)
            plt.show()
        if 'rl' in sel:
            fig, (ax1,ax2) = plt.subplots(1,2)
            fig.suptitle(f'RL recover - {iter} iterations')
            ax1.set_title('Before')
            field_image(fig,ax1,field,**kwargs)
            ax2.set_title('After')
            field_image(fig,ax2,Sr1,**kwargs)
            plt.show()
            kwargs['title'] = 'Image - mean value'
            new_pic = Sr1-mean_val
            fast_image(np.where(new_pic<0, 0, new_pic),**kwargs)    
    return Sr

def mask_size(recover_field: NDArray, field: NDArray | None = None, display_fig: bool = False, **kwargs) -> int:
    tmp_field = recover_field.copy()
    dim = len(tmp_field)
    size = 0
    cond = np.array([False]*4)
    diff = np.zeros(4)
    for i in range(dim-1):
        if not cond[0]: 
            diff[0] = tmp_field[i+1,i+1] - tmp_field[i,i]
        if not cond[1]: 
            diff[1] = tmp_field[dim-1-i-1,dim-1-i-1] - tmp_field[dim-1-i,dim-1-i]
        if not cond[2]: 
            diff[2] = tmp_field[i+1,dim-1-i-1] - tmp_field[i,dim-1-i]
        if not cond[3]: 
            diff[3] = tmp_field[dim-1-i-1,i+1] - tmp_field[dim-1-i,i]
        cond = diff <= 0
        if all(cond): 
            size = i
            break
    print('Lim Cut',size)
    size0 = size
    diff = np.zeros((4,dim-size-1))
    for i in range(size+1,dim-1):
        diff[0] = tmp_field[size+1:,i+1] - tmp_field[size+1:,i]
        diff[1] = tmp_field[size+1:,dim-1-i-1] - tmp_field[size+1:,dim-1-i]
        diff[2] = tmp_field[i+1,size+1:] - tmp_field[i,size+1:]
        diff[3] = tmp_field[dim-1-i-1,size+1:] - tmp_field[dim-1-i,size+1:]
        if diff.max() >= 0:
            size = 4*i//3
            break
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
            a_size = grad_check(tmp_field,(x,y),mean_val,size=size,acc=acc)
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
        a_size = grad_check(tmp_field,index,mean_val,size=size,acc=acc)
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
