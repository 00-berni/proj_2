import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from .display import fast_image
from .field import N, noise, Uniform, Gaussian, check_field

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
        if ymax == 0: results += [0]
        else: ycond = lambda yval, ylim: yval < ylim 
    elif 'by' in direction:
        yd = -1
        ymax = min(size, y)
        if ymax == 0: results += [0]
        else: ycond = lambda yval, ylim: yval < ylim 
    
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
    if len(results) == 1: results = results[0] 
    return results


def grad_check(field: np.ndarray, index: tuple[int,int], size: int = 3) -> tuple[np.ndarray,np.ndarray]:
    x,y = index
    yf = moving('fy',field,index,size)
    xf = moving('fx',field,index,size)
    yb = moving('by',field,index,size)
    xb = moving('bx',field,index,size)
    xff, yff = moving('fxfy',field,index,size)
    xbb, ybb = moving('bxby',field,index,size)
    xfb, yfb = moving('fxby',field,index,size)
    xbf, ybf = moving('bxfy',field,index,size)
    
    print('x',x)
    print('y',y)
    print('xf',xf,xff,xfb)
    print('yf',yf,yff,yfb)
    print('xb',xb,xbb,xbf)
    print('yb',yb,ybb,ybf)
    xf = min(xff,xf,xfb)
    yf = min(yff,yf,ybf)
    xb = min(xbb,xb,xbf)
    yb = min(ybb,yb,yfb)

    X = slice(x-xb,x+xf+1)
    Y = slice(y-yb,y+yf+1)
    
    fast_image(field[X,Y],v=1)

def size_est(field: np.ndarray, index: tuple[int,int], thr: float = 1e-3, size: int = 3) -> tuple:
    """Estimation of the size of the object
    The function takes in input the most luminous point, calls the `grad_check()` function
    to investigate the presence of other nearby objects and then estimates the size of the
    target conditionated by the choosen threshold value.

    :param field: field matrix
    :type field: np.ndarray
    :param index: coordinates of the most luminous point
    :type index: tuple[int,int]
    :param thr: threshold to get the size of the element, defaults to 1e-3
    :type thr: float
    :param size: the upper limit for the size of the obj, defaults to 3
    :type size: int, optional

    :return: a tuple with the size in each directions
    :rtype: tuple
    """
    # coordinates of the object
    x, y = index
    # saving the value in that position
    max_val = field[index]
    # getting the frame in which studying the size
    limits, ind_limits = grad_check(field,index,size)
    """
        Looking for free directions is done
        because the purpose is to investingate 
        the direction for which the frame has 
        the maximum size.
    """
    # condition for at least one free direction
    if ind_limits[0] != -1:
        # taking the maximum size in free direction group
        pos = max(limits[ind_limits])
        # storing the index for that direction
        ind_pos = np.where(limits == pos)[0][0]
    # if there is none, one takes the maximum size
    else:
        # taking the maximum size 
        ind_pos = int(np.argmax(limits))
        # storing its index
        pos = limits[ind_pos]
    
    # creating the parameter for the size definition by threshold
    ratio = 1
    # inizializing the index to explore the field
    i = 0
    # condition to move along x direction
    if ind_pos < 2:
        # direction for the exploration
        sign = (-2*ind_pos + 1)
        # taking pixels until the threshold or the edge 
        while(ratio > thr and i < pos):
            i += 1
            # uploading the parameter     
            ratio = field[x+sign*i,y]/max_val
    # condition to move along y direction
    else:
        # direction for the exploration
        sign = (-2*ind_pos + 5)
        # taking pixels until the threshold or the edge 
        while(ratio > thr and i < pos):
            i += 1     
            # uploading the parameter     
            ratio = field[x,y+sign*i]/max_val
    # saving estimated width
    width = i
    # taking the min between width and size from grad_check()
    return tuple(min(width, w) for w in limits)

def zero_check(field: np.ndarray) -> np.ndarray:
    cut_field = np.copy(field)
    cut_xind = []
    cut_yind = []
    
    xind, yind = np.where(cut_field == 0.)        
   
    values, counts = np.unique(xind, return_counts=True)
    double_pos = np.where(counts>1)[0]
    if len(double_pos) != 0:
        cut_xind += [val for val in values[double_pos]]
    del values, counts, double_pos
  
    values, counts = np.unique(yind, return_counts=True)
    double_pos = np.where(counts>1)[0]
    if len(double_pos) != 0:
        cut_yind += [val for val in values[double_pos]]
    del values, counts, double_pos

    if len(cut_xind) != 0:
        cut_field = np.delete(cut_field,cut_xind,axis=0)    
    if len(cut_yind) != 0:
        cut_field = np.delete(cut_field,cut_yind,axis=1)

    xdim, ydim = cut_field.shape
    if xdim != 1 and ydim != 1:
        ind = np.where(cut_field == 0.)
        if len(ind) != 0:
            cut_xind = []
            cut_yind = []
            xind = ind[0]
            xcnt = 0
            pos = []
            for i in range(len(xind)):
                if (xdim - xcnt) == 1:
                    break
                elif (xdim - xcnt) % 2 == 0:
                    cut_xind += [xind[i]]
                    xcnt += 1
                    pos += [i]
            yind = ind[1]
            ycnt = 0
            for i in range(len(yind)):
                if (ydim - ycnt) == 1:
                    break
                elif (ydim - ycnt) % 2 == 0:
                    if i not in pos:
                        cut_yind += [yind[i]]
                        ycnt += 1
            
            if len(cut_xind) != 0:
                cut_field = np.delete(cut_field,cut_xind,axis=0)    
            if len(cut_yind) != 0:
                cut_field = np.delete(cut_field,cut_yind,axis=1)
    
    xdim, ydim = cut_field.shape
    if xdim != 1 and ydim != 1:
        ind = np.where(cut_field == 0.)
        if len(ind) != 0:
            if xdim > 1 :
                xind = ind[0]
                cut_field = np.delete(cut_field,xind,axis=0)
            if ydim > 1:
                yind = ind[1]
                cut_field = np.delete(cut_field,yind,axis=1)
    return cut_field

##*
def object_isolation(field: np.ndarray, obj: tuple[int,int], coord: list[tuple], thr: float = 1e-3, size: int = 3) -> np.ndarray | None:
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
    # coordinates of central object
    x, y = obj
    # calculating the size of the object
    wx_u, wx_d, wy_u, wy_d = size_est(field, obj, thr=thr, size=size)
    # extracting the obj
    extraction = field[x - wx_d : x + wx_u +1, y - wy_d : y + wy_u +1].copy()
    extraction = zero_check(extraction)
    # removing the object from the field
    field[x - wx_d : x + wx_u +1, y - wy_d : y + wy_u +1] = 0.0
    # removing the obj from the available points in the field
    for k in [(x+i, y+j) for i in range(-wx_d,wx_u+1) for j in range(-wy_d, wy_u+1)]:
        # control condition
        if k in coord: coord.remove(k)
    # returning the extracted obj
    return extraction if extraction.shape != (0,) else None

def objects_detection(field: np.ndarray, dark_noise: float, back: float, thr: float = 1e-1, size: int = 3, coord: list[tuple] = [], loop_num: int = 4, r_field: bool = False, display: bool = True) -> tuple[list, np.ndarray] | list[np.ndarray]:
    """Extracting stars from field
    The function calls the `object_isolation()` function iteratively until
    the SNR (`snr`) is less than 2. Then it returns a list that contains 
    the extracted objects.

    :param field: field matrix
    :type field: np.ndarray
    :param dark_noise: threshold for consider a signal
    :type dark_noise: float    
    :param thr: threshold for the size of an obj, defaults to 1e-3
    :type thr: float, optional
    :param size: max size of an obj, defaults to 3
    :type size: int, optional
    :param coord: list of possible positions in the field, defaults to []
    :type coord: list[tuple], optional
    :param point_num: number of points to draw, defaults to 100
    :type point_num: int, optional
    :param loop_num: number of loops over which one want to average, defaults to 4
    :type loop_num: int, optional

    :return: list of extracted objects
    :rtype: list[np.ndarray]
    """
    # coping the field to preserve it
    tmp_field = field.copy()
    # saving size of the field
    dim = len(tmp_field)
    # creating an empty list to store the extracted objects
    a_extraction = []
    # generating list with all possible position, if it was not 
    if len(coord) == 0:  
        coord = [(i,j) for i in range(dim) for j in range(dim)]
    # evaluating the maximum in the field
    max_pos = np.unravel_index(np.argmax(tmp_field, axis=None), tmp_field.shape)
    max_val = tmp_field[max_pos]
    # averaging between n0 and noise from dark
    n0 = max(back,dark_noise)

    # evaluating the first SNR and storing it
    snr = max_val / n0
    # counter
    cnt = 1
    # starting the loop 
    while snr > 1 and cnt < loop_num:
        extraction = object_isolation(tmp_field, max_pos, coord, thr, size)
        if extraction is not None:
            # appending the new extracted object to the list
            a_extraction += [extraction]
            cnt += 1
        # evaluating the new maximum in the field
        max_pos = np.unravel_index(np.argmax(tmp_field, axis=None), tmp_field.shape)
        max_val = tmp_field[max_pos]
        snr = max_val/n0
    # displaying the image
    if display:
        plt.figure()
        plt.title('Field after extraction')
        plt.imshow(tmp_field,norm='log',cmap='gray')
        plt.colorbar()
        plt.show()
    
    if r_field:
        return a_extraction, tmp_field
    else:
        # returning list with objects
        return a_extraction

        