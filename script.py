import numpy as np
import skysimulation.field as field
import skysimulation.restoration as restore

if __name__ == '__main__':
    figure = False
    n = 3e-4
    F_bsd, F, F_b, F_bs = field.field_builder(100,500,display_fig=figure,results='Fbs')
    dark = restore.dark_elaboration(field.Uniform(n),display_fig=figure)
    Fn = F_bsd - dark
    back = restore.bkg_est(Fn,figure)
    print(f'Extimated background maxval:\t{10**back}')
    back = 10**back
    field.fast_image(F_bsd)
    extlist = restore.objects_detection(F_bsd,dark.max(),back)
    for elem in extlist:
        field.fast_image(elem,v=1)

