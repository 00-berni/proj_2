import numpy as np
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

    objnum = 5

    extr = restore.object_isolation(F_bsd,max(back,dark.max()),objnum=objnum)
    print('\n\nRESHAPE')
    field.fast_image(F_bsd)
    extr = restore.object_isolation(F_bsd,max(back,dark.max()),objnum=objnum,reshape=True,reshape_corr=True)