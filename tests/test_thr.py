"""
    Test file for the optimization of the stopping method
"""

##* Packages
import numpy as np
import matplotlib.pyplot as plt
from test_func import *
import os

# dimension of the matrix
N = int(1e2+1)
# number of stars
M = int(1e2)


def find_max(field):
    return np.unravel_index(np.argmax(field), field.shape)

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
            * ind_neg : indeces in `a_size` for free directions
    :rtype: tuple[np.ndarray,np.ndarray]
    """    
    # field size
    dim = len(field)
    # object position
    x, y = index
    # treatment for edges; f stays for forward, b for backward
    xlim_f, ylim_f = min(size, dim-2-x), min(size, dim-2-y)
    xlim_b, ylim_b = min(size, x+1), min(size, y+1)
    # limits for the object
    xsize_f0, xsize_b0 = min(size, dim-1-x), min(size, x)
    ysize_f0, ysize_b0 = min(size, dim-1-y), min(size, y)
    # saving them in an array
    # the negative sign is used to take trace of free direction
    a_size0 = np.array([xsize_f0,xsize_b0,ysize_f0,ysize_b0], dtype=int) * -1
    # creating an array to store the size of the object in differt directions
    a_size = np.copy(a_size0)
    # moving in the four directions
    for i_f, i_b, j_f, j_b in zip(range(xlim_f), range(xlim_b), range(ylim_f), range(ylim_b)):
        # studying the sign of the gradient
        a_size[0] = i_f if (field[x+i_f+1, y]-field[x+i_f, y] >= 0 and a_size[0] == a_size0[0]) else a_size[0]
        a_size[1] = i_b if (field[x-i_b-1, y]-field[x-i_b, y] >= 0 and a_size[1] == a_size0[1]) else a_size[1]
        a_size[2] = j_f if (field[x, y+j_f+1]-field[x, y+j_f] >= 0 and a_size[2] == a_size0[2]) else a_size[2]
        a_size[3] = j_b if (field[x, y-j_b-1]-field[x, y-j_b] >= 0 and a_size[3] == a_size0[3]) else a_size[3]
        # diagonal
        a_size[0], a_size[2] = (i_f, j_f) if (field[x+i_f+1, y+j_f+1]-field[x+i_f, y+j_f] >= 0 and a_size[0] == a_size0[0] and a_size[2] == a_size0[2]) else  (a_size[0], a_size[2])
        a_size[0], a_size[3] = (i_f, j_b) if (field[x+i_f+1, y-j_b-1]-field[x+i_f, y-j_b] >= 0 and a_size[0] == a_size0[0] and a_size[3] == a_size0[3]) else  (a_size[0], a_size[3])
        a_size[1], a_size[2] = (i_b, j_f) if (field[x-i_b-1, y+j_f+1]-field[x-i_b, y+j_f] >= 0 and a_size[1] == a_size0[1] and a_size[2] == a_size0[2]) else  (a_size[1], a_size[2])
        a_size[1], a_size[3] = (i_b, j_b) if (field[x-i_b-1, y-j_b-1]-field[x-i_b, y-j_b] >= 0 and a_size[1] == a_size0[1] and a_size[3] == a_size0[3]) else  (a_size[1], a_size[3])
        # when for every direction there's an obj, the for cycle stops 
        if (True in (a_size == a_size0)) == False: break
    # looking for free direction
    condition = np.where(a_size < 0)[0]
    # if there is at least one
    if len(condition) != 0:
        # saving the indices
        ind_neg = condition
        # removing the sign
        a_size[ind_neg] *= -1
    # if there is none
    else:
        # storing the information
        ind_neg = np.array([-1])
    return a_size, ind_neg

"""
#  COSE DA AGGIUSTARE
#   X 1. capire come aggiustare la situazione quando 2 oggetti sono appiccicati
#   X 2. capire come tenere di conto del bordo
""" 
def approx_width(field: np.ndarray, index: tuple[int,int], thr: float = 1e-10, size: int = 3) -> tuple:
    """Extimation of the size of the object
    The function takes in input the most luminous point, calls the `grad_check()` function
    to investigate the presence of other objects and then to extimate the size of the
    target conditionated by the choosen threshold value.

    :param field: field matrix
    :type field: np.ndarray
    :param index: coordinates of the most luminous point
    :type index: tuple
    :param thr: threshold to get the size of the element, defaults to 1e-10
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
    # getting the size of the object
    limits, ind_limits = grad_check(field,index,size)
    # condition for at least one free direction
    if ind_limits[0] != -1:
        # taking the maximum size in free direction group
        pos = max(limits[ind_limits])
        # storing the index for that direction
        ind_pos = np.where(limits == pos)[0][0]
    else:
        # taking the maximum size 
        ind_pos = int(np.argmax(limits))
        # storing its index
        pos = limits[ind_pos]
    # creating the parameter for the comparison
    ratio = 1
    # inizializing the index to explore the field
    i = 0
    # moving along x direction
    if ind_pos < 2:
        # direction for the exploration
        sign = (-2*ind_pos + 1)
        # take pixels until the threshold or the adge 
        while(ratio > thr and i < pos):
            i += 1
            # upload the parameter     
            ratio = field[x+sign*i,y]/max_val
    # moving along y direction
    else:
        # direction for the exploration
        sign = (-2*ind_pos + 5)
        # take pixels until the threshold 
        # take pixels until the threshold or the adge 
        while(ratio > thr and i < pos):
            i += 1     
            # upload the parameter     
            ratio = field[x,y+sign*i]/max_val
    # saving extimated width
    width = i
    # taking the min between width and size
    return tuple(min(width,i) for i in limits)

def object_isolation(obj: tuple[int,int], field: np.ndarray, coord: list[tuple], thr: float = 1e-10, size: int = 3) -> np.ndarray:
    """To isolate the most luminous star object.
    The function calls the `approx_width()` function 
    to extract from the field the object of interest.

    :param field: field matrix
    :type field: np.ndarray
    :param thr: threshold for `approx_width()` function, defaults to 1e-10
    :type thr: float, optional
    :param dim: size of the field, defaults to `N`
    :type dim: int, optional
    
    :return: the object matrix
    :rtype: np.ndarray
    """
    # calculating the size of the object
    wx_u, wx_d, wy_u, wy_d = approx_width(field, obj, thr=thr, size=size)
    # extracting the coordinates
    x, y = obj
    # printing infos
    # print(f'star ({x},{y}) val {field[obj]} -> {wx_u}, {wx_d}, {wy_u}, {wy_d}')
    # isolating the obj and coping in order to preserve the field matrix
    extraction = field[x - wx_d : x + wx_u +1, y - wy_d : y + wy_u +1].copy()
    # removing the object from the field
    field[x - wx_d : x + wx_u +1, y - wy_d : y + wy_u +1] = 0.0
    # removing the obj from the available points 
    for k in [(x+i, y+j) for i in range(-wx_d,wx_u+1) for j in range(-wy_d, wy_u+1)]:
        # control condition
        if k in coord: coord.remove(k)
    # #! DA RIMUOVERE PRIMA DI MANDARE !#
    # if M < 11:
    #     plt.imshow(extraction, norm='log')
    #     plt.colorbar()
    #     plt.show()
    #     plt.imshow(field, norm='log')   
    #     plt.colorbar()
    #     plt.show()
    # #!                               !#
    return extraction

##*
def evaluate_noise(field: np.ndarray, coord: list[tuple], point_num: int = 100, loop_num: int = 4) -> float:
    """To estimate the background noise value.
    The function draws points in the field and averages over them in order to get an estimation of the mean luminosity

    :param field: field matrix
    :type field: np.ndarray
    :param coord: list of possible position in the field
    :type coord: list[tuple]
    :param point_num: number of points to draw, defaults to 100
    :type point_num: int, optional
    :param loop_num: number of loops over which one want to average, defaults to 4
    :type loop_num: int, optional
    :return: the estimated noise value
    :rtype: float
    """
    # defining the variable for the estimated noise
    est_noise = 0
    # saving the size of coord
    dim = len(coord)
    # the number of points depends on the number of remained points in the field
    n_point = min(point_num, dim)
    # making `loop_num` drawing over which average
    for i in range(loop_num):
        # drawing positions in the coordinates list
        ind = np.random.choice(dim, size=n_point, replace=False)
        # making an array for the drawn elements
        element = np.array([field[coord[i]] for i in ind])
        # evaluating the mean
        est_noise += sum(element)/len(element)
    # averaging
    est_noise /= loop_num
    return est_noise

##*
def counting_stars(field: np.ndarray, dark_noise: float, thr: float = 1e-3, size: int = 3, coord: list[tuple] = [], point_num: int = 100, loop_num: int = 4, diag: int = 0) -> list[np.ndarray]:
    """Extracting stars from field
    The function calls the `object_isolation()` function iteratively until
    the signal-to-noise ratio (`s_t_n`) is less than 2

    :param field: field matrix
    :type field: np.ndarray
    :param dark_max: threshold for consider a signal
    :type dark_max: float
    :param thr: threshold for the size of an obj, defaults to 1e-3
    :type thr: float, optional
    :param size: max size of an obj, defaults to 3
    :type size: int, optional

    :return: the list of extracted object. Its len() is the number of found obj
    :rtype: list[np.ndarray]
    """
    # coping the field to preserve it
    tmp_field = field.copy()
    # saving size of the field
    dim = len(tmp_field)
    # creating an empty list to store the extracted objects
    a_extraction = []
    # list for positions
    a_pos = []
    # evaluating the maximum in the field
    max_pos = np.unravel_index(np.argmax(tmp_field, axis=None), tmp_field.shape)
    max_val = tmp_field[max_pos]
    a_pos += [max_pos]
    # appending the new extracted object to the list
    a_extraction += [object_isolation(max_pos, tmp_field, coord, thr, size)]
    # printing it
    if diag==1: print(f'max = {max_val}')
    # generating list with all possible position, if it was not 
    if len(coord) == 0:  
        coord = [(i,j) for i in range(dim) for j in range(dim)]
    # Before the search of objects, let's set an initial value for the noise, 
    #   that will set the condition to start the estimation of the signal-to-noise ratio
    # first noise value
        # appending the new extracted object to the list
    n0 = evaluate_noise(tmp_field, coord, point_num=100, loop_num=loop_num)
    # averaging with the noise from dark
    n0 = (n0 + dark_noise) / 2
    # taking the minimum between n and the 10% of maximum luminosity
    n0 = min(n0, max_val/10)
    if diag==1: print(f'noise =\t{n0}')
    #! for me to control
    i = 0
    # evaluating the signal-to-noise ratio
    s_t_n = max_val / n0
    if diag==1: print(f'stn =\t{1/s_t_n*100:.2f} %')
    # starting the loop 
    while s_t_n > 2 and i < M+3:
        # evalueting the new maximum in the field
        max_pos = np.unravel_index(np.argmax(tmp_field, axis=None), tmp_field.shape)
        max_val = tmp_field[max_pos]
        a_extraction += [object_isolation(max_pos, tmp_field, coord, thr, size)]        
        a_pos += [max_pos]
        if diag==1: print(f'max = {max_val}')
        # condition to start the noise evaluation
        if s_t_n <= 50:
            n0 = evaluate_noise(field,coord,point_num,loop_num)
            # taking the max between n and noise from dark
            n0 = max(n, dark_noise)
        s_t_n = max_val/n0
        if diag==1:
            print(f'noise =\t{n0}')
            print(f'stn =\t{1/s_t_n*100:.2f} %')
        i += 1
        if i == M+2: print('!WARNING!: ITERATION EXCEDED!')
    # plt.imshow(tmp_field, norm='log', cmap='gray_r')
    # plt.show()
    return a_extraction, a_pos



if __name__ == '__main__':
    # Current directory
    pwd = os.getcwd()
    det_noise = 3e-4
    n = 0.2/1e2

    test_field = np.zeros((N,N))
    number = 1000
    lum = [1e-3,n+1e-3]*number +  [n*5] * (number//3) + [n*10]*5
    from numpy.random import randint
    print(lum[:4])
    org_pos = []
    pos = (randint(0,N),randint(0,N))
    for l in lum:
        while test_field[pos] != 0:
            pos = (randint(0,N),randint(0,N))
        test_field[pos] = l
        org_pos += [pos]
    noise_camp = atm_seeing(noise(n) + test_field) + noise(det_noise)

    d = dark_elaboration(det_noise)
    d = d[np.unravel_index(np.argmax(d), d.shape)]
    print(n,d)


    #noise_camp = noise_camp - d

    # plt.figure(1)

    # plt.imshow(test_field, norm='log', cmap='gray_r')
    # # plt.colorbar()

    # plt.figure(2)
    # plt.imshow(noise_camp, norm='log', cmap='gray_r')

    # making a list with every position in the field, format (x,y)
    grid = [(i,j) for i in range(N) for j in range(N)]
    # creating an empty array to collect the maximum values
    maxvalues = np.array([])
    # setting the number of points taken around
    size_num = 100
    # setting the number of iteration for the drawing
    cnt = 4
    # noise
    pm_noise = 0
    for ii in range(cnt):
        # drawing positions in the grid
        ind = np.random.choice(len(grid), size=size_num, replace=False)
        # making an array for the drawn elements
        element = np.array([noise_camp[grid[i]] for i in ind])
        mean = sum(element)/len(element)
        # evaluating the mean
        pm_noise += mean
    # averaging
    pm_noise /= cnt
    ss = f'pm0 = {pm_noise}'
    pm_noise = (d + pm_noise)/2
    ss2 = f'pm1 = {pm_noise}'
    pm_noise = min(pm_noise, noise_camp[find_max(noise_camp)]/10)
    # setting the number of points taken around
    size_num = 700
    # setting the noise thr to d of the detector
    thr = d
    # setting the starting max value
    maxval = 1
    # initializing the iteration counter
    it = 0
    # good counter
    pos_cnt = 0

    # diagnostic parameter
    diag = 0

    # starting the loop
    while maxval/thr > 10 and it < len(lum)+5:
        if diag==1: print(f'iteration {it}')
        # finding the maximum
        maxind = np.unravel_index(np.argmax(noise_camp), noise_camp.shape)
        maxval = noise_camp[maxind]
        xm, ym = maxind
        if maxval == 0:
            print('!WARNING: MAXVAL == 0!')
            break
        if maxind in org_pos:
            pos_cnt += 1
        # removing the obj
        edges = [min(3,xm), min(4,N-xm), min(3,ym), min(4,N-ym)]
        noise_camp[xm-edges[0]:xm+edges[1], ym-edges[2]:ym+edges[3]] = 0.0
        # saving the maxval
        maxvalues = np.append(maxvalues,maxval)
        if diag==1:
            print(f'max =\t{maxval}')
            print(f'pos =\t{maxind}')
        # removing the maxval obj from the grid
        for k in [(xm-3+i,ym-3+j) for i in range(7) for j in range(7)]:
            # removing only coordinates which aren't have been removed yet
            if k in grid:
                grid.remove(k)
        # saving the actual dim of grid
        dim = len(grid)
        # the number of points depends on the number of remained points in the field
        n_point = min(size_num, dim)
        if maxval/pm_noise<=50:
            # defining the variable to mean the noise
            reco_n_bg = 0
            hold = []
            for ii in range(cnt):
                # drawing positions in the grid
                ind = np.random.choice(dim, size=n_point, replace=False)
                # making an array for the drawn elements
                element = np.array([noise_camp[grid[i]] for i in ind])
                mean = sum(element)/len(element)
                # evaluating the mean
                reco_n_bg += mean
                hold += [mean]
            # averaging
            reco_n_bg /= cnt
            # controlling the value of noise
            thr = max(reco_n_bg, d)
            if diag==1:
                print(f'noise =\t{reco_n_bg}')
                print(f'great =\t{max(hold)}')
                print(f'thr =\t{thr}')
                print(f's:n =\t{maxval/thr}')
                print(f's:n2 =\t{maxval/max(hold)}')
        it += 1
        if it == len(lum)+1: print(f'MAX ITERATION NUMBER EXCEDED! -> {it}')

    found = len(maxvalues)
    wanted = number//3 + 5
    expected = wanted + number        
    print('-----------'+ f'\nFound\tWanted\tExpected\n{found}\t{wanted}\t{expected}')
    print(f'Count: {pos_cnt}\nMiscount: {found-pos_cnt}')
    print(f'Fraction\nf/e =\t{found/expected*100:.0f} %\nf/tot =\t{found/(wanted+number*2)*100 :.0f} %')
    if pos_cnt != found: print(f'c/e =\t{pos_cnt/expected*100:.0f} %\nc/tot =\t{pos_cnt/(wanted+number*2)*100 :.0f} %')
    print(f'Accuracy: {pos_cnt/found*100:.2f} %')
    # plt.figure(3)
    # plt.imshow(noise_camp, norm='log', cmap='gray_r')
    # plt.colorbar()
    print(ss)
    print(ss2)
    print(f'pm = {pm_noise}')

    # print(listpos)
    # print(reco_n_bg)


    plt.show()
    # from scipy.signal import argrelextrema

    stars = [5,10,50,70,100,200,500,1000,2000,3000]
    # stars = [3,4,5,6,7,8] 
    # stars = [10,20,40,50,70,80,90] 
    # stars = [100,200,400,500,700,800,900] 


    a_fake = []
    a_prec = []


    for M in stars:
        
        print('\n==========\n'+f'Results for {M} stars')
        # generation of the field and the stars
        F, S = initialize(sdim=M)

        # taking the 0.2% of the luminosity of a 1 solar mass star
        # for 1 solar mass star L = M**beta = 1 in solar luminosity unit 
        n = 0.2/1e2

        objs = S.lum
        objs = np.where(objs > n)[0]
        objs = len(objs)

        # add background noise
        F_n = F + noise(n)

        # detector noise
        det_noise = 3e-4

        # generate atmosferic seeing image without sky noise
        F_s = atm_seeing(F, sigma=0.5)

        # generate atmosferic seeing image with sky noise
        F_sn = atm_seeing(F_n, sigma=0.5)

        # plt.imshow(F_sn, norm='log', cmap='gray_r')
        # plt.colorbar()
        # plt.show()
        # add detector noise, set to 3e-4 (> than background one)
        F_sn += noise(det_noise)

        ##* Detection
        """ 
            For the recostruction of psf it's usefull only one star, 
            for exemple the most luminous one.
        """
        #test_field = np.zeros((N,N))
        test_field = F_sn.copy()
        # find the maximum value coordinates
        max_ind = np.unravel_index(np.argmax(test_field, axis=None), test_field.shape)

        m_x, m_y = max_ind

        # mean dark
        dark = dark_elaboration(det_noise)
        # for the noise estimated from dark take the maximum
        d = dark[np.unravel_index(np.argmax(dark), dark.shape)]
        print(f'dark =\t{d}')

        # number of points
        p_num = 200 
        # thr to define an obj
        thr = 1e-1
        a_extraction, a_pos = counting_stars(test_field, d, thr=thr, size=3, point_num=p_num)
        cnt = 0
        star_pos = [(i,j) for i,j in zip(S.x,S.y)]
        for k in a_pos:
            if k in star_pos: cnt+=1
        
        miss = cnt-len(a_extraction)

        a_fake += [miss/len(a_extraction)*100]
        a_prec += [len(a_extraction)/objs*100 if objs!=0 else 0]

        print(f'\nThe star number is\nExt\tTrue\n{len(a_extraction)}\t{M}')
        print(f'ext/tot: {len(a_extraction)/M*100:.2f} %')
        print(f'Expected: {objs}')
        print(f'exp/tot: {objs/M*100:.2f} %')
        print(f'Precision: {len(a_extraction)/objs*100:.2f} %')
        print(f'Good: {cnt}\t{cnt/len(a_extraction)*100:.2f} %')
        print(f'Miscounts: {miss}')
        print(f'Fake objs: {miss/len(a_extraction)*100:.2f} %\n')

        selected = a_extraction[-1]
        thr_tmp = selected[np.unravel_index(np.argmax(selected), selected.shape)]



        # plt.imshow(np.where(F_s>thr_tmp, F_s, 0.0), norm='log', cmap='gray_r')
        # plt.imshow(F_s, norm='log')
    
    a_fake = np.array(a_fake)
    a_prec = np.array(a_prec)

    xx = np.arange(len(stars))
    
    plt.figure(1)
    plt.title('Misleading objects')
    plt.hlines(0,min(xx),max(xx),'gray','dashdot',alpha=0.7)
    plt.plot(xx, a_fake, '--.', label='true - extracted')
    plt.xticks(xx,stars)
    plt.xlabel('number of stars')
    plt.ylabel('(true - extracted) / extracted [%]')
    plt.grid()

    plt.figure(2)
    plt.title('Precision: fraction of expected')
    plt.hlines(100,min(xx),max(xx),'gray','dashdot',alpha=0.7)
    plt.plot(xx, a_prec, '--.', label='ext/exp')
    plt.xticks(xx,stars)
    plt.xlabel('number of stars')
    plt.ylabel('extracted / expected [%]')
    plt.grid()
    

    plt.show()
