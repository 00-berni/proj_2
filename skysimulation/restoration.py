import numpy as np
import matplotlib.pyplot as plt
from .field import N, noise

##*
def dark_elaboration(n_value: float = 3e-4, iteration: int = 3, dim: int = N) -> np.ndarray:
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
    dark = noise(n_value, dim=dim)
    # making the loop
    for i in range(iteration-1):
        dark += noise(n_value, dim=dim)
    # averaging
    dark /= iteration
    return dark

##*
def grad_check(field: np.ndarray, index: tuple[int,int], size: int = 3) -> tuple[np.ndarray,np.ndarray]:
    """Function explores the neighbourhood of a selected object and gives its size.
    It studies the gradient around the obj in the four cardinal directions and the diagonal ones.
    It takes in account also the presence of the edges of the field.
    If no other obj is found in a whichever direction, that one is stored and returned.

    :param field: the field matrix
    :type field: np.ndarray
    :param index: the obj coordinates
    :type index: tuple[int,int]
    :param size: the upper limit for the size of the obj, defaults to 3
    :type size: int, optional
    
    :return: a tuple with:
             * a_size : array with the size of the obj in each directions, like [x_up, x_down, y_up, y_down]
             * ind_neg : indeces in `a_size` for free directions or in case of no free direction np.array([-1])
    :rtype: tuple[np.ndarray,np.ndarray]
    """    
    # field size
    dim = len(field)
    # object coordinates
    x, y = index
    # treatment for edges; f stays for forward, b for backward
    # maximum dimension of the object (the frame)
    xsize_f0, xsize_b0 = min(size, dim-1-x), min(size, x)
    ysize_f0, ysize_b0 = min(size, dim-1-y), min(size, y)
    # saving them in an array
    # the negative sign is used to take trace of free direction
    # starting from isolate object case
    a_size0 = np.array([xsize_f0,xsize_b0,ysize_f0,ysize_b0], dtype=int) * -1
    # limits of the frame to compute the gradient
    xlim_f, ylim_f = min(size, dim-2-x), min(size, dim-2-y)
    xlim_b, ylim_b = min(size, x+1), min(size, y+1)
    # creating an array to store the size of the object in differt directions
    a_size = np.copy(a_size0)
    # moving in the eight directions
    for i_f, i_b, j_f, j_b in zip(range(xlim_f), range(xlim_b), range(ylim_f), range(ylim_b)):
        """
        The algorithm is simple:  
            if the chosen direction is free and the trend is not 
            monotonic from the studying pixel on, routine stops
            and pixel distance from the center is stored
        """  
        # studying the sign of the gradient
        # along the x direction
        a_size[0] = i_f if ((field[x+i_f+1, y]-field[x+i_f, y] >= 0 or field[x+i_f+1, y] == 0) and a_size[0] == a_size0[0]) else a_size[0]
        a_size[1] = i_b if ((field[x-i_b-1, y]-field[x-i_b, y] >= 0 or field[x-i_b+1, y] == 0) and a_size[1] == a_size0[1]) else a_size[1]
        # along the y direction
        a_size[2] = j_f if ((field[x, y+j_f+1]-field[x, y+j_f] >= 0 or field[x, y+j_f] == 0) and a_size[2] == a_size0[2]) else a_size[2]
        a_size[3] = j_b if ((field[x, y-j_b-1]-field[x, y-j_b] >= 0 or field[x, y-j_b] == 0) and a_size[3] == a_size0[3]) else a_size[3]
        # along diagonal directions
        a_size[0], a_size[2] = (i_f, j_f) if ((field[x+i_f+1, y+j_f+1]-field[x+i_f, y+j_f] >= 0 or field[x+i_f+1, y+j_f+1] == 0) and a_size[0] == a_size0[0] and a_size[2] == a_size0[2]) else  (a_size[0], a_size[2])
        a_size[0], a_size[3] = (i_f, j_b) if ((field[x+i_f+1, y-j_b-1]-field[x+i_f, y-j_b] >= 0 or field[x+i_f+1, y-j_b-1] == 0) and a_size[0] == a_size0[0] and a_size[3] == a_size0[3]) else  (a_size[0], a_size[3])
        a_size[1], a_size[2] = (i_b, j_f) if ((field[x-i_b-1, y+j_f+1]-field[x-i_b, y+j_f] >= 0 or field[x-i_b-1, y+j_f+1] == 0) and a_size[1] == a_size0[1] and a_size[2] == a_size0[2]) else  (a_size[1], a_size[2])
        a_size[1], a_size[3] = (i_b, j_b) if ((field[x-i_b-1, y-j_b-1]-field[x-i_b, y-j_b] >= 0 or field[x-i_b-1, y-j_b-1] == 0) and a_size[1] == a_size0[1] and a_size[3] == a_size0[3]) else  (a_size[1], a_size[3])
        # if no free direction is present, 
        # there is no reason to run again the loop
        if (True in (a_size == a_size0)) == False: break
    """ 
        To compute the following extraction of the object
        from the field, the knowing of the presence of
        free directions is needed    
    """
    # looking for free direction
    condition = np.where(a_size < 0)[0]
    # if there is at least one
    if len(condition) != 0:
        # saving the indices
        ind_neg = condition
        # removing the negative sign
        a_size[ind_neg] *= -1
    # if there is none
    else:
        # storing the information
        ind_neg = np.array([-1])
    return a_size, ind_neg

##*
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

##*
def object_isolation(field: np.ndarray, obj: tuple[int,int], coord: list[tuple], thr: float = 1e-3, size: int = 3) -> np.ndarray:
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
    # removing the object from the field
    field[x - wx_d : x + wx_u +1, y - wy_d : y + wy_u +1] = 0.0
    # removing the obj from the available points in the field
    for k in [(x+i, y+j) for i in range(-wx_d,wx_u+1) for j in range(-wy_d, wy_u+1)]:
        # control condition
        if k in coord: coord.remove(k)
    # returning the extracted obj
    return extraction

##*
def evaluate_noise(field: np.ndarray, coord: list[tuple], point_num: int = 100, loop_num: int = 4, step0: int = 0) -> float:
    """To estimate the background noise value.
    The function draws points in the field and averages over them 
    in order to get an estimation of the mean background luminosity

    :param field: field matrix
    :type field: np.ndarray
    :param coord: list of possible positions in the field
    :type coord: list[tuple]
    :param point_num: number of points to draw, defaults to 100
    :type point_num: int, optional
    :param loop_num: number of loops over which one want to average, defaults to 4
    :type loop_num: int, optional
    :param step0: parameter for the first step
    :type step0: int    

    :return: the estimated noise value
    :rtype: float
    """
    # saving the size of coord
    dim = len(coord)
    # the number of points depends on the number of remained points in the field
    n_point = min(point_num, dim)
    # defining the variable for the estimated noise
    est_noise = []
    # making `loop_num` drawing over which average
    for i in range(loop_num):
        # drawing positions in the coordinates list
        ind = np.random.choice(dim, size=n_point, replace=False)
        # making an array for the drawn elements
        element = np.array([field[coord[i]] for i in ind])
        # storing the mean
        est_noise += [sum(element)/len(element)]
    # estimating the mean noise according to the `step0` value
    est_noise = max(est_noise) if step0 == 0 else sum(est_noise)/len(est_noise)
    return est_noise

##*
def objects_detection(field: np.ndarray, dark_noise: float, thr: float = 1e-1, size: int = 3, coord: list[tuple] = [], point_num: int = 100, loop_num: int = 4) -> list[np.ndarray]:
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
    """
        Before searching objects, an initial value for the noise is estimated.
        This value will set the start condition for the MNER.
    """
    # first estimation of noise    
    n0 = evaluate_noise(tmp_field, coord, point_num=100, loop_num=loop_num, step0=1)
    # averaging between n0 and noise from dark
    n0 = (n0 + dark_noise) / 2
    # taking the minimum between n0 and the 10% of maximum luminosity of the field
    n0 = min(n0, max_val/10)

    # evaluating the first SNR and storing it
    snr0 = max_val / n0
    # initializing the SNR variable
    snr = snr0
    # starting the loop 
    while snr > 2:
        # appending the new extracted object to the list
        a_extraction += [object_isolation(tmp_field, max_pos, coord, thr, size)]
        # evaluating the new maximum in the field
        max_pos = np.unravel_index(np.argmax(tmp_field, axis=None), tmp_field.shape)
        max_val = tmp_field[max_pos]
        # condition to start the MNER
        if snr0 <= 50:
            # estimation of noise
            n = evaluate_noise(tmp_field,coord,point_num,loop_num)
            # taking the max between n and noise from dark
            n = max(n, dark_noise)
            # estimating the new SNR
            snr = max_val/n
        # MNER does not start
        else:
            # computing the new SNR0
            snr0 = max_val/n0
            # updating the SNR
            snr = snr0
    # displaying the image
    plt.figure()
    plt.title('Field after extraction')
    plt.imshow(tmp_field,norm='log',cmap='gray')
    plt.colorbar()
    plt.show()
    # returning list with objects
    return a_extraction

