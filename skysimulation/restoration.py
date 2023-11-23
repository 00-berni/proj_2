import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from .display import fast_image
from .field import Gaussian, N, Uniform, check_field, noise


def peak_pos(field: np.ndarray) -> int | tuple[int,int]:
    if len(field.shape) == 1:
        return field.argmax()
    else:
        return np.unravel_index(field.argmax(),field.shape)
##*
def dark_elaboration(params: tuple[str, float | tuple], iteration: int = 3, dim: int = N, display_fig: bool = False) -> np.ndarray:
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
    dark = noise(params, dim=dim)
    # making the loop
    for i in range(iteration-1):
        dark += noise(params, dim=dim)
    # averaging
    dark /= iteration
    if display_fig:
        fast_image(dark,title=f'Dark elaboration\nAveraged on {iteration} iterations')
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
def object_isolation(field: np.ndarray, thr: float, size: int = 3, objnum: int = 10, reshape: bool = False, reshape_corr: bool = False, display_fig: bool = False) -> np.ndarray | None:
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
            if len(pos[0]) != 0:
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
            if display_fig: fast_image(obj) 
            k += 1
    fast_image(tmp_field)

    if len(extraction) == 0: extraction = None
    return extraction

def kernel_fit(obj: np.ndarray, back: float, noise: float) -> float:
    dim = len(obj)
    m = np.arange(dim)
    x, y = np.meshgrid(m,m)
    c = dim // 2
    x -= c
    y -= c
    r = np.sqrt(x**2 + y**2)
    sigma0 = 1
    print('sigma',sigma0)
    k0 = obj.max()

    def fit_func(x,sigma,k):
        return k * Gaussian(sigma).value(x) + back + noise
    
    err = (back+noise)*np.ones(obj.shape)
    from scipy.optimize import curve_fit
    initial_values = [sigma0,k0]
    pop, pcov = curve_fit(fit_func,r.flatten(),obj.flatten(),initial_values,sigma=err.flatten())
    sigma, k = pop
    Dsigma, _ = np.sqrt(pcov.diagonal())
    print(f'sigma = {sigma} +- {Dsigma}')
    chi_sq = (((obj-fit_func(r,sigma,k))/err)**2).sum()
    chi0 = len(obj.flatten()) - 2
    print(f'chi_sq = {chi_sq/chi0*100:.2f} +- {np.sqrt(2/chi0)*100:.2f} %')
    return sigma

def kernel_extimation(extraction: list[np.ndarray], back: float, noise: float, dim: int, display_plot: bool = False, all_results: bool = False) -> np.ndarray | tuple[np.ndarray,tuple[float,float]]:
    a_sigma = np.array([],dtype=float)
    for obj in extraction:
        sigma = kernel_fit(obj,back,noise)
        a_sigma = np.append(a_sigma,sigma)
        del sigma
    sigma = np.mean(a_sigma)
    Dsigma = np.sqrt(((sigma-a_sigma)**2).sum()/(len(a_sigma)*(len(a_sigma)-1)))
    print(f'\nsigma = {sigma:.5f} +- {Dsigma:.5f}')
    kernel = Gaussian(sigma)
    kernel = kernel.kernel(dim)
    if display_plot:
        fast_image(kernel)
    
    if all_results:
        return kernel,(sigma,Dsigma)
    else:
        return kernel



def LR_deconvolution(field: np.ndarray, kernel: np.ndarray, back: float, noise: float, iter: int = 17) -> np.ndarray:
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
