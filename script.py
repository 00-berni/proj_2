import numpy as np
import matplotlib.pyplot as plt
import skysimulation.field as field
import skysimulation.restoration as restore

if __name__ == '__main__':
    N = 100
    M = 4500
    figure = False
    norm = 'log'
    back = field.BACK_PARAM
    det = field.NOISE_PARAM
    S, I = field.field_builder(N,M,back_param=back,det_param=det,display_fig=figure,norm=norm)
    S.plot_info(field.ALPHA,field.BETA)
    n = restore.bkg_est(I,True)
    print(n/field.K)