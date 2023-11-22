import numpy as np
import matplotlib.pyplot as plt
import skysimulation.field as field
import skysimulation.restoration as restore

if __name__ == '__main__':
    figure = False
    n = 3e-4
    F_bsd, F, F_b, F_bs = field.field_builder(100,1000,display_fig=figure,results='Fbs')
    dark = restore.dark_elaboration(field.Uniform(n),display_fig=figure)
    Fn = F_bsd - dark
    back = restore.bkg_est(Fn,figure)
    print(f'Extimated background maxval:\t{10**back}')
    back = 10**back
    print(F.shape,F_bsd.shape)
    field.fast_image(F_bsd)

    objnum = 10

    ndet = dark.max()

    # extr = restore.object_isolation(F_bsd,max(back,dark.max()),objnum=objnum)
    print('\n\nRESHAPE')
    field.fast_image(F_bsd)
    extr = restore.object_isolation(F_bsd,max(back,dark.max()),objnum=objnum,reshape=True,reshape_corr=True)
    ext_kernel = restore.kernel_extimation(extr,back,ndet,len(F),True)
    F_ext = restore.LR_deconvolution(F_bsd,ext_kernel,back,ndet)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(F_bsd,norm='log')
    plt.colorbar()
    plt.subplot(1 ,2,2)
    plt.imshow(F_ext,norm='log')
    plt.colorbar()
    plt.show()