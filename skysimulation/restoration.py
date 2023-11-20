import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from .display import fast_image
from .field import N, noise, Uniform, Gaussian, check_field

def peak_pos(field: np.ndarray) -> int | tuple[int,int]:
    if len(field.shape) == 1:
        return field.argmax()
    else:
        return np.unravel_index(field.argmax(),field.shape)
##*
def dark_elaboration(distr: Uniform | Gaussian, iteration: int = 3, dim: int = N, display_fig: bool = False) -> np.ndarray:
    """The function computes a number (`iteration`) of darks
    and averages them in order to get a mean estimation 
    of the detector noise

    :param n_value: detector noise, defaults to 3e-4
    :type n_value: float, optional
    :param iteration: number of darks to compute, defaults to 3
    :type iteration: int, optional

    :return: mean dark
    :rtype: np.ndarray
    """
    # generating the first dark
    dark = noise(distr, dim=dim)
    # making the loop
    for i in range(iteration-1):
        dark += noise(distr, dim=dim)
    # averaging
    dark /= iteration
    if display_fig:
        fast_image(dark,v=1,title=f'Dark elaboration\nAveraged on {iteration} iterations')
    return dark

def bkg_est(field: np.ndarray, display_fig: bool = False) -> float:
    field = np.copy(field).flatten()
    field = field[np.where(field > 0)[0]]
    num = np.sqrt(len(field))
    bins = np.arange(np.log10(field).min(),np.log10(field).max(),1/num)
    counts, bins = np.histogram(np.log10(field),bins=bins)
    tmp = counts[counts.argmax()+1:]
    dist = abs(tmp[:-2] - tmp[2:])
    pos = np.where(counts == tmp[dist.argmax()+2])[0]
    mbin = (max(bins[pos])+bins[counts.argmax()])/2
    if display_fig:
        plt.figure(figsize=(14,10))
        plt.stairs(counts, bins,fill=True)
        # plt.axvline(max(bins[pos]),0,1,linestyle='--',color='orange')
        # plt.axvline(bins[counts.argmax()],0,1,linestyle='--',color='red')
        plt.axvline(mbin,0,1,linestyle='--',color='orange')
        plt.xlabel('$\\log_{10}(F_{sn})$')
        plt.ylabel('counts')
        plt.show()
    return mbin

def moving(direction: str, field: np.ndarray, index: tuple[int,int], size: int = 3) -> list[int]:
    tmp_field = field.copy()
    dim = len(tmp_field)
    x, y = index
    # size += 1
    
    results = []
    xd = 0
    yd = 0
    xmax = x
    ymax = y
    xcond = lambda xval, xlim: xlim == x
    ycond = lambda yval, ylim: ylim == y

    if 'fx' in direction:
        print('hey')
        xd = 1
        xmax = min(size, dim-1-x)
        if xmax == 0: results += [0]
        else: xcond = lambda xval, xlim: xval < xlim 
    elif 'bx' in direction:
        xd = -1
        xmax = min(size, x)
        if xmax == 0: results += [0]
        else: xcond = lambda xval, xlim: xval < xlim 
    if 'fy' in direction:
        yd = 1
        ymax = min(size, dim-1-y)
        print(ymax)
        if ymax == 0: 
            # if len(results) == 1:
            #     results = [[0,0]]
            # else: results += [0]
            results += [0]
        else: ycond = lambda yval, ylim: yval < ylim 
    elif 'by' in direction:
        yd = -1
        ymax = min(size, y)
        if ymax == 0: 
            # if len(results) == 1:
            #     results = [[0,0]]
            # else: results += [0]
            results += [0]
        else: ycond = lambda yval, ylim: yval < ylim 
    print('1 result',results)
    if len(results) == 0:
        xsize = 0
        ysize = 0
        condition = xcond(xsize,xmax) and ycond(ysize,ymax)
        print(xcond(xsize,xmax),xmax)
        while condition:
            step0 = tmp_field[x + xsize*xd, y + ysize*yd]
            step1 = tmp_field[x + (xsize+1)*xd, y + (ysize+1)*yd]
            grad = step1 - step0
            if grad >= 0 or step1 == 0:
                print(grad,step1)
                print(xsize,ysize)
                break
            else:
                xsize += 1
                ysize += 1
                condition = xcond(xsize,xmax) and ycond(ysize,ymax)
        if 'x' in direction: results += [xsize]
        if 'y' in direction: results += [ysize]
    elif len(results) == 1:
        if 'x' in direction and 'y' in direction:
            results = [0,0]        
    if len(results) == 1: results = results[0] 
    print('2 result',results)
    return results


def grad_check(field: np.ndarray, index: tuple[int,int], size: int = 3) -> tuple[np.ndarray,np.ndarray]:
    mov = lambda val: moving(val,field,index,size)
    xy_dir = ['fxfy','fxby','bxfy','bxby'] 
    a_xysize = np.array([mov(dir) for dir in xy_dir])
    print('sizes',a_xysize)
    xf_size = a_xysize[:2,0]
    xb_size = a_xysize[2:,0]
    yf_size = a_xysize[(0,2),1]
    yb_size = a_xysize[(1,3),1]

    a_size = np.array([[[mov('fx'),*xf_size],[mov('bx'),*xb_size]],
                       [[mov('fy'),*yf_size],[mov('by'),*yb_size]]])
    x_size, y_size = a_size.min(axis=2)
    return x_size, y_size
    

##*
def object_isolation(field: np.ndarray, thr: float, size: int = 3, objnum: int = 10, reshape: bool = False, reshape_corr: bool = False) -> np.ndarray | None:
    """To isolate the most luminous star object.
    The function calls the `size_est()` function to compute the size of the object and
    then to extract it from the field.

    :param field: field matrix
    :type field: np.ndarray
    :param obj: object coordinates
    :type obj: tuple[int,int]
    :param coord: list of possible positions in the field
    :type coord: list[tuple]
    :param thr: threshold for `size_est()` function, defaults to 1e-3
    :type thr: float, optional
    :param size: the upper limit for the size of the obj, defaults to 3
    :type size: int, optional
    
    :return: the extracted object matrix
    :rtype: np.ndarray
    """
    tmp_field = field.copy()

    extraction = []
    
    k = 0 
    while k < objnum:
        # finding the peak
        index = peak_pos(tmp_field)
        peak = tmp_field[index]
        # checking the value
        if peak <= thr:
            break
        # computing size
        x, y = index
        a_size = grad_check(field,index,size)
        print('a_size',a_size)
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
        print('SLices: ',xr,yr)
        obj = field[xr,yr].copy() 
        tmp_field[xr,yr] = 0.0
        print('a_size 2',a_size)
        if all([0,0] == a_size[0]) or all([0,0] == a_size[1]):
            print(f'Remove obj: ({x},{y})')
        else: 
            if reshape and reshape_corr:
                if 0 in [xu,xd,yu,yd]:
                    xpad_pos = (0,0)
                    ypad_pos = (0,0)
                    if xu == 0: xpad_pos = (0,xd)
                    elif xd == 0: xpad_pos = (xu,0)
                    if yu == 0: ypad_pos = (0,yd)
                    elif yd == 0: ypad_pos = (yu,0)
                    print('PAd',xpad_pos,ypad_pos)
                    obj = np.pad(obj,(xpad_pos,ypad_pos),'reflect')
            extraction += [obj]
            fast_image(obj,v=1) 
            k += 1
        fast_image(tmp_field,v=1)

    if len(extraction) == 0: extraction = None
    return extraction

def kernel_fit(obj: np.ndarray):
    dim = len(obj)
    m = np.arange(dim)
    x, y = np.meshgrid(m,m)
    c = dim // 2
    x -= c
    y -= c
    r = np.sqrt(x**2 + y**2)

    diff1 = abs(obj[c+1,c]-obj[c-1,c])
    diff2 = abs(obj[c,c+1]-obj[c,c-1])
    sigma0 = (diff1+diff2)/2
    k0 = obj.max()

    def fit_func(x,sigma,k):
        return k * Gaussian(sigma).value(x)
   
    from scipy.optimize import curve_fit
    initial_values = [sigma0,k0]
    pop, pcov = curve_fit(fit_func,r.flatten(),obj.flatten(),initial_values)
    print(pop)
    