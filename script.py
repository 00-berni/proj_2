import numpy as np
import matplotlib.pyplot as plt
import skysimulation.field as field
import skysimulation.restoration as restore

if __name__ == '__main__':
    N = 100
    M = 4500
    figure = False
    norm = 'linear'
    back = field.BACK_PARAM
    det = field.NOISE_PARAM

    print('\n--- Build the Field ---')
    S, I = field.field_builder(N,M,back_param=back,det_param=det,display_fig=figure,norm=norm)
    S.plot_info(field.ALPHA,field.BETA)

    print('\n--- Dark ---')
    dark = restore.dark_elaboration(det,dim=N)

    print('\n--- Background Estimation ---')
    bkg = restore.bkg_est(I,True)
    print(bkg/field.K)

    print('\n--- Objects Extraction ---')
    obj = restore.object_isolation(I,max(bkg,dark.mean()),size=7,reshape=True,reshape_corr=True,display_fig=True,norm=norm)

    print('\n--- Kernel Estimation ---')
    # kernel = restore.kernel_estimation()