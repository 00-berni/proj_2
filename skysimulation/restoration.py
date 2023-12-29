import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from .display import fast_image, field_image
from .field import Gaussian, N, Uniform, noise


def peak_pos(field: np.ndarray) -> int | tuple[int,int]:
    """Finding the coordinate/s of the maximum

    :param field: array
    :type field: np.ndarray

    :return: coordinate/s
    :rtype: int | tuple[int,int]
    """
    if len(field.shape) == 1:
        return field.argmax()
    else:
        return np.unravel_index(field.argmax(),field.shape)
##*
def dark_elaboration(params: tuple[str, float | tuple], iteration: int = 3, dim: int = N, display_fig: bool = False, **kwargs) -> np.ndarray:
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
    :rtype: np.ndarray
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

def bkg_est(field: np.ndarray, display_fig: bool = False) -> float:
    """Estimating a value for the background

    :param field: the field
    :type field: np.ndarray
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

def moving(direction: str, field: np.ndarray, index: tuple[int,int], back: float, size: int = 3, acc: float = 1e-5) -> list[int]:
    """Looking in one direction

    `direction` is a string and contains two parameters:

      1. `'f'` or `'b'` mean forward or backward, respectively.
      2. `'x'` or `'y'` mean horizontal or vertical direction, respectively.

    It is also possible to combine different directions such as `'fxby'`
    (combinations along same axis are not allowed) 

    :param direction: selected direction
    :type direction: str
    :param field: the field
    :type field: np.ndarray
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


def grad_check(field: np.ndarray, index: tuple[int,int], back: float, size: int = 3) -> tuple[np.ndarray,np.ndarray]:
    """Checking the gradient trend around an object

    :param field: the field
    :type field: np.ndarray
    :param index: coordinates of the object
    :type index: tuple[int,int]
    :param size: the maximum size of the object, defaults to 3
    :type size: int, optional
    
    :return: size of the object (x,y)
    :rtype: tuple[np.ndarray,np.ndarray]
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
    
def selection(objs: list[np.ndarray], apos: np.ndarray, size: int, maxdist: int = 5) -> tuple[list[np.ndarray], list[np.ndarray], list[None | np.ndarray]]:
    """Selecting the objects for the fit

    :param objs: list of extracted objects
    :type objs: list[np.ndarray]
    :param apos: positions array
    :type apos: np.ndarray
    :param size: max selected size
    :type size: int
    :param maxdist: max accepted distance between objects, defaults to 5
    :type maxdist: int, optional
    :return: list of the selected and rejected objects 
    :rtype: tuple[list[np.ndarray], list[None | np.ndarray]]
    """
    #: method to compute the length
    dist = lambda x,y: np.sqrt(x**2 + y**2)
    # extracting positions
    x = np.copy(apos[:,0])
    y = np.copy(apos[:,1])
    # computing distances
    adist = np.array( [dist(x[i]-x, y[i]-y) for i in range(len(x))] )
    # initializing variables
    del_obj = []
    xdel = np.array([])
    ydel = np.array([])
    sel_obj = [*objs]
    a_objs = np.array( [np.copy(obj) for obj in objs] , dtype='object' )
    maxdist += size
    pos = np.where(np.logical_and(adist < maxdist, adist != 0))
    dim = len(pos)
    if dim != 0:
        pos = np.array(pos)
        # print('adist',adist)
        print('pos',pos)
        pos = np.unique(pos[0,:])
        dim = len(pos)
        if dim % 2 == 0:
            mid = dim//2-1
            pos = pos[:mid:-1]
            print('del',(x[pos],y[pos]))
            # pos = pos[:mid]
            print('pos_cut',pos)
            del_obj = [a_objs[i] for i in pos]
            a_objs = np.delete(a_objs,pos,axis=0)
            xdel = np.append(xdel,x[pos])
            ydel = np.append(ydel,y[pos])
            x = np.delete(x,pos)
            y = np.delete(y,pos)
            del mid
    del pos,dim

    if size != 1:
        tmp_sizes = np.array([len(obj) for obj in a_objs])
        pos = np.where(tmp_sizes <= 3)[0]
        if len(pos) != 0:
            del_obj += [np.copy(a_objs[i]) for i in pos]
            a_objs = np.delete(a_objs,pos,axis=0)
            xdel = np.append(xdel,x[pos])
            ydel = np.append(ydel,y[pos])
            x = np.delete(x,pos)
            y = np.delete(y,pos)
        del pos
    
    if len(a_objs) > 1:
        pos = np.array([],dtype='int')
        for k in range(len(a_objs)):
            obj = a_objs[k]
            xmax,ymax = peak_pos(obj)
            print(f'Max positions:\n\t{xmax} and {ymax}')
            if xmax != len(obj)//2 or ymax != len(obj)//2:
                print(f'! The {k} object is uncorrect !')
            lim = 5 if len(obj) > 10 else len(obj)//2 
            for i in range(lim):
                i += 1
                r = obj[xmax+i,ymax]
                l = obj[xmax-i,ymax]
                u = obj[xmax,ymax+i]
                d = obj[xmax,ymax-i]
                diff1 = abs(r/l - u/d) 
                diff2 = abs(r/d - u/l)
                if diff1 >= 0.2 or diff2 >= 0.2 or (diff1 > 0.16 and diff2 > 0.16):
                    print(f'\tDifferences:\n\t{diff1}\t{diff2}')
                    pos = np.append(pos,k)
                    break
        print(pos,len(pos))
        if len(pos) != 0:
            del_obj += [np.copy(a_objs[i]) for i in pos]
            a_objs = np.delete(a_objs,pos,axis=0)
            xdel = np.append(xdel,x[pos])
            ydel = np.append(ydel,y[pos])
            x = np.delete(x,pos)
            y = np.delete(y,pos)
    sel_obj = list(a_objs)
    return sel_obj, [x,y], del_obj, [xdel,ydel]


##*
def object_isolation(field: np.ndarray, thr: float, size: int = 3, objnum: int = 10, reshape: bool = False, reshape_corr: bool = False, sel_cond: bool = False, display_fig: bool = False,**kwargs) -> np.ndarray | None:
    """To isolate the most luminous star object.
    The function calls the `size_est()` function to compute the size of the object and
    then to extract it from the field.

    :param field: the field
    :type field: np.ndarray
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
    :rtype: np.ndarray | None
    """
    tmp_field = field.copy()
    display_field = field.copy()

    extraction = []
    a_pos = []
    
    if display_fig: 
        tmp_kwargs = {key: kwargs[key] for key in kwargs.keys() - {'title'}} 

    k = 0 
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
        a_size = grad_check(field,index,thr,size)
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
        if all([0,0] == a_size[0]) or all([0,0] == a_size[1]):
            print(f'Remove obj: ({x},{y})')
        else: 
            remove_cond = True 
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
                else: remove_cond = False
            
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
            
            if remove_cond:
                display_field[xr,yr] = 0.0
            
            extraction += [obj]
            a_pos += [[x,y]]


            if display_fig: 
                tmp_kwargs['title'] = f'N. {k+1} object {index}'
                fast_image(obj,**tmp_kwargs) 
            k += 1
    a_pos = np.array(a_pos)
    
    if sel_cond and len(extraction) > 1:
        asel,aselpos, adel, adelpos = selection(extraction,a_pos,size)
        print(':: Results of selection ::')
        print(len(extraction),len(asel),len(adel))
        if len(adel) != 0:
            for elem in adel:
                fast_image(elem,title='Removed object',**kwargs)

    if 'title' not in kwargs:
        kwargs['title'] = 'Field after extraction'
    fast_image(display_field,**kwargs)
    fast_image(tmp_field,**kwargs)
    if sel_cond and len(extraction) > 1:
        fig, ax = plt.subplots(1,1)
        kwargs.pop('title',None)
        field_image(fig,ax,display_field,**kwargs)
        ax.plot(adelpos[1],adelpos[0],'.',color='red',label='removed objects')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1,1)
        kwargs.pop('title',None)
        field_image(fig,ax,field,**kwargs)
        ax.plot(adelpos[1],adelpos[0],'.',color='red',label='removed objects')
        ax.plot(aselpos[1],aselpos[0],'.',color='blue',label='chosen objects')
        ax.legend()
        plt.show()

    if len(extraction) == 0: extraction = None
    print(':: End ::')
    return extraction

def kernel_fit(obj: np.ndarray, back: float, noise: float) -> float:
    """Estimating the sigma of the kernel

    :param obj: extracted objects
    :type obj: np.ndarray
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
    c = dim // 2
    x -= c
    y -= c
    r = np.sqrt(x**2 + y**2)
    sigma0 = 1
    print('sigma',sigma0)
    k0 = obj.max()

    def fit_func(x,sigma,k):
        return k * Gaussian(sigma).value(x)
    
    err = np.sqrt(back**2 + noise**2)*np.ones(obj.shape)
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

def kernel_estimation(extraction: list[np.ndarray], back: float, noise: float, dim: int, all_results: bool = False, display_plot: bool = False, **kwargs) -> np.ndarray | tuple[np.ndarray,tuple[float,float]]:
    """Estimation of the kernel from a Gaussian model

    :param extraction: extracted objects
    :type extraction: list[np.ndarray]
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
    :rtype: np.ndarray | tuple[np.ndarray,tuple[float,float]]
    """
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
        if 'title' not in kwargs:
            kwargs['title'] = 'Estimated kernel'
        fast_image(kernel,**kwargs)
    
    if all_results:
        return kernel,(sigma,Dsigma)
    else:
        return kernel


def LR_deconvolution(field: np.ndarray, kernel: np.ndarray, back: float, noise: float, iter: int = 17) -> np.ndarray:
    """Richardson-Lucy deconvolution algorithm

    :param field: the field
    :type field: np.ndarray
    :param kernel: estimated kernel
    :type kernel: np.ndarray
    :param back: estimated value of the background
    :type back: float
    :param noise: mean noise
    :type noise: float
    :param iter: number of iterations, defaults to 17
    :type iter: int, optional
    
    :return: recostructed field
    :rtype: np.ndarray
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
