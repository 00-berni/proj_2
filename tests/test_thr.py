"""
	SECOND EXERCISE: PSF PROJECT

	Author:	Bernardo Vettori
	Date:	
	References:
			- Shore, S.N., "The Tapestry of Modern Astrophysics"
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

    plt.figure(1)
    plt.imshow(test_field, norm='log', cmap='viridis')
    # plt.colorbar()

    plt.figure(2)
    plt.imshow(noise_camp, norm='log')

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
    # starting the loop
    while maxval/thr > 5 and it < len(lum)+5:
        print(f'iteration {it}')
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
            print(f'noise =\t{reco_n_bg}')
            print(f'great =\t{max(hold)}')
            # controlling the value of noise
            thr = max(reco_n_bg, d)
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
    plt.figure(3)
    plt.imshow(noise_camp, norm='log')
    # plt.colorbar()
    print(ss)
    print(ss2)
    print(f'pm = {pm_noise}')

    # print(listpos)
    # print(reco_n_bg)


    plt.show()