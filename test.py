"""
TEST SCRIPT
===========

    The aim of the script is to test the function `object_check()` of the
    `restoration.py` package.

    The test consists in 2 steps:
      1. Generate a single object with some background
      2. Apply the method to the object and show the result
    The script does it for different value of the background mean

"""

import numpy as np
import matplotlib.pyplot as plt
import skysimulation.stuff as stf
import skysimulation.field as fld
import skysimulation.restoration as rst
from skysimulation.display import fast_image, field_image

### MAIN ###
if __name__ == '__main__':
    dim = 13
    field0 = np.zeros((dim,dim))
    field0[dim//2, dim//2] = 100
    fast_image(field0, 'uninitialized')
    bkg_distr = lambda mean : stf.Gaussian(sigma=mean * 20e-2, mu=mean)
    ker_sigma = 3
    kernel = stf.Gaussian(ker_sigma).kernel()
    mean_noise = 0.5
    for mean_bkg in [0,5,10,20,30,50]:
        bkg = bkg_distr(mean_bkg)
        field = field0 + np.sqrt(bkg.field(shape=field0.shape)**2)
        fast_image(field,'Obj before convolution')
        field = stf.field_convolve(field, kernel, bkg) #+ stf.Gaussian(sigma=mean_noise * 50e-2, mu=mean_noise).field(field0.shape) - mean_noise
        fast_image(field,f'The object for background of {mean_bkg}')
        index = stf.peak_pos(field)
        result = rst.object_check(field, index, mean_bkg, None)
        if result is not None:
            obj, index, dim = result
            fast_image(obj,'Object is fine')
        else:
            fast_image(field,'Object is not fine')

        
