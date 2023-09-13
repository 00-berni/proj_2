"""
    Test file for the optimization of the stopping method
"""

##* Packages
import os.path as ph
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJ_FOLDER = ph.split(ph.dirname(ph.realpath(__file__)))[0]

sys.path.insert(1, ph.join(PROJ_FOLDER))
from skysimulation.field import *
from skysimulation.restoration import dark_elaboration, object_isolation
# from test_func import *

# dimension of the matrix
N = int(1e2+1)
# number of stars
M = int(1e2)


def find_max(field):
    return np.unravel_index(np.argmax(field), field.shape)


##*
def evaluate_noise(field: np.ndarray, coord: list[tuple], point_num: int = 100, loop_num: int = 4, step0: int = 0) -> float:
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
    est_noise = []
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
        est_noise += [sum(element)/len(element)]
    # averaging
    est_noise = max(est_noise) if step0 == 0 else sum(est_noise)/len(est_noise) 
    return est_noise

##*
def counting_stars(field: np.ndarray, dark_noise: float, thr: float = 1e-3, size: int = 4, coord: list[tuple] = [], point_num: int = 100, loop_num: int = 4, diag: int = 0, m: int = M) -> list[np.ndarray]:
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
    a_extraction += [object_isolation(tmp_field, max_pos, coord, thr, size)]
    # printing it
    if diag==1: print(f'max = {max_val}')
    # generating list with all possible position, if it was not 
    if len(coord) == 0:  
        coord = [(i,j) for i in range(dim) for j in range(dim)]
    # Before the search of objects, let's set an initial value for the noise, 
    #   that will set the condition to start the estimation of the signal-to-noise ratio
    # first noise value
        # appending the new extracted object to the list
    n0 = evaluate_noise(tmp_field, coord, point_num=100, loop_num=loop_num, step0=1)
    # averaging with the noise from dark
    n0 = (n0 + dark_noise) / 2
    # taking the minimum between n and the 10% of maximum luminosity
    n0 = min(n0, max_val/10)
    if diag==1: print(f'noise0 =\t{n0}')
    #! for me to control
    i = 0
    # evaluating the signal-to-noise ratio
    s_t_n0 = max_val / n0
    s_t_n = s_t_n0
    if diag==1: print(f'stn0 =\t{1/s_t_n*100:.2f} %')
    # starting the loop 
    while s_t_n > 2 and i < m+3:
        # evaluating the new maximum in the field
        max_pos = np.unravel_index(np.argmax(tmp_field, axis=None), tmp_field.shape)
        max_val = tmp_field[max_pos]
        a_extraction += [object_isolation(tmp_field, max_pos, coord, thr, size)]        
        a_pos += [max_pos]
        if diag==1: print(f'max = {max_val}')
        if s_t_n0 <= 50:
            nn = evaluate_noise(tmp_field,coord,point_num,loop_num)
            # taking the max between n and noise from dark
            nn = max(nn, dark_noise)
            s_t_n = max_val/nn
            if diag==1:
                print(f'noise =\t{nn}')
                print(f'stn =\t{1/s_t_n*100:.2f} %')
        else:
            s_t_n0 = max_val/n0
            s_t_n = s_t_n0
            if diag==1:
                print(f'noise0 =\t{n0}')
                print(f'stn0 =\t{1/s_t_n*100:.2f} %')

        i += 1
        if i == m+2: print('!WARNING!: ITERATION EXCEDED!')
    if diag == 1:
        plt.imshow(tmp_field, norm='log', cmap='viridis')
        plt.colorbar()
        plt.show()
    return a_extraction, a_pos



if __name__ == '__main__':
    
    # Current directory
    pwd = os.getcwd()

    diag = 0

    
    # noises
    n = 0.2/1e2         # background noise 
    det_noise = 3e-4    # detector noise
    
    # n_num = [1,3,10,25,50,70,100,120,170,340,560,700]
    # n_num = [2,5,10,25,50,100,250,400,500]
    n_num = [5, 20, 80]

    n_objs = []

    n_extr = []
    n_fake = []

    for number in n_num: 
        # generating the field
        test_field = np.zeros((N,N))
        
        # generating the sample

        lum = [n/10,n+2e-3]*number +  [n*5] * (number//2) + [n*10] * (number //3) + [n*1e2]
        # number of stars
        nstars = len(lum)
        # saving number of objs
        n_objs += [nstars]
        # number of expected object
        nexp = number + (number // 2) + (number // 3) + 1
        print(f'\n=====\nRun for {nstars} stars\n')
        print(f'Detector noise:\t{det_noise}\nBackground:\t{n}\n')
        print(f'Stars:\nLum\tNum\n{n*1e2}\t1\n{n*10}\t{number // 3}\n{n*5}\t{number//2}\n{n+1e-3}\t{number}\n{n/10}\t{number}\n')

        # list to store the positions of stars
        org_pos = []
        # making a list with every position in the field, format (x,y)
        grid = [(i,j) for i in range(N) for j in range(N)]
        # locating the stars
        ind = np.random.choice(len(grid), size=nstars, replace=False)
        for i in range(nstars):
            x,y = grid[ind[i]]
            test_field[x,y] = lum[i]
            org_pos += [(x,y)]

        # generating the field
        noise_camp = atm_seeing(noise(n) + test_field) + noise(det_noise)

        # elaboration of dark
        d = dark_elaboration(det_noise)
        d = d[np.unravel_index(np.argmax(d), d.shape)]

        # making a list with every position in the field, format (x,y)
        grid = [(i,j) for i in range(N) for j in range(N)]
        # creating an empty array to collect the maximum values
        maxvalues = np.array([])
        # setting the number of points taken around
        size_num = 200
        # thr to define an obj
        thr = 1e-2
        a_extraction, a_pos = counting_stars(noise_camp, d, thr=thr, size=3, point_num=size_num, m=nstars, diag=diag)

        cnt = 0
        for k in a_pos:
            if k in org_pos: cnt+=1

        # number of extracted
        next = len(a_extraction)

        print(f'====\nExtracted objects:\t{next}\nFraction over tot:\t{next/nstars*100:.2f} %\nExpected objects:\t{nexp}\nFraction over exp:\t{next/nexp*100:.2f} %\n')

        n_extr += [next/nexp*100] 

        fake_obj = next - cnt

        print(f'True objects:\t{cnt}\nFake objects:\t{fake_obj}\nFraction over ext objs:\t{cnt/next*100:.2f} %\n====')

        n_fake += [cnt/next*100]

    # print(f'\n=====\nStudied objs:\t{len(n_objs)}')

    xx = np.arange(len(n_objs))

    n_extr = np.array(n_extr)
    n_fake = np.array(n_fake)

    plt.figure(1)
    plt.subplots_adjust(hspace=0)
    plt.subplot(2,1,1)
    plt.title('Precision and accuracy for set stars')
    plt.hlines(100,min(xx),max(xx),'gray','dashdot',alpha=0.7)
    plt.plot(xx, n_extr, '--.', label='Precision')
    plt.xticks(xx,[])
    plt.ylabel('extracted / expected [%]')
    plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    plt.hlines(100,min(xx),max(xx),'gray','dashdot',alpha=0.7)
    plt.plot(xx,n_fake, '--.', color='red', label='Accuracy')
    plt.xticks(xx,n_objs)
    plt.xlabel('number of stars')
    plt.ylabel('true / extracted [%]')
    plt.grid()
    plt.legend()
    
    plt.show()


    print('\n\n*****************\n')


    # stars = [5,10,20,50,100,200,500,800,1000]
    # stars = [5,10,50,70,100,200,500,1000,2000]
    stars = [3,4,5,6,7,8] 
    # stars = [10,20,40,50,70,80,90] 
    # stars = [100,200,400,500,700,800,900] 


    a_fake = []
    a_prec = []


    for m in stars:
        
        print('\n==========\n'+f'Results for {m} stars')
        # generation of the field and the stars
        F, S = initialize(sdim=m)

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
        thr = 1e-2
        a_extraction, a_pos = counting_stars(test_field, d, thr=thr, size=3, point_num=p_num, diag=diag, m=m)
        cnt = 0
        x, y = S.pos
        star_pos = [(i,j) for i,j in zip(x,y)]
        for k in a_pos:
            if k in star_pos: cnt+=1
        
        miss = cnt-len(a_extraction)

        a_fake += [cnt/len(a_extraction)*100]
        a_prec += [len(a_extraction)/objs*100 if objs!=0 else 0]

        print(f'\nThe star number is\nExt\tTrue\n{len(a_extraction)}\t{m}')
        print(f'ext/tot: {len(a_extraction)/m*100:.2f} %')
        print(f'Expected: {objs}')
        print(f'exp/tot: {objs/m*100:.2f} %')
        print(f'Precision: {len(a_extraction)/objs*100:.2f} %')
        print(f'Good: {cnt}\t{cnt/len(a_extraction)*100:.2f} %')
        print(f'miscounts: {miss}')
        print(f'Accuracy: {cnt/len(a_extraction)*100:.2f} %\n')

        selected = a_extraction[-1]
        thr_tmp = selected[np.unravel_index(np.argmax(selected), selected.shape)]



        # plt.imshow(np.where(F_s>thr_tmp, F_s, 0.0), norm='log', cmap='gray_r')
        # plt.imshow(F_s, norm='log')
    
    a_fake = np.array(a_fake)
    a_prec = np.array(a_prec)

    xx = np.arange(len(stars))
    
    plt.figure(1)
    plt.subplots_adjust(hspace=0)
    plt.subplot(2,1,1)
    plt.title('Precision and accuracy for stars generated from IMF')
    plt.hlines(100,min(xx),max(xx),'gray','dashdot',alpha=0.7)
    plt.plot(xx, a_prec, '--.', label='Precision')
    plt.xticks(xx,[])
    plt.ylabel('extracted / expected [%]')
    plt.legend()
    plt.grid()

    plt.subplot(2,1,2)
    plt.hlines(100,min(xx),max(xx),'gray','dashdot',alpha=0.7)
    plt.plot(xx, a_fake, '--.', color='red', label='Accuracy')
    plt.xticks(xx,stars)
    plt.xlabel('number of stars')
    plt.ylabel('true  / extracted [%]')
    plt.legend()
    plt.grid()
    

    plt.show()
