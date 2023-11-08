import numpy as np
import skysimulation.field as field
import skysimulation.restoration as restore
from skysimulation.display import fast_image

if __name__ == '__main__':
    figure = False
    n = 3e-4
    F_bsd, F_b, F_bs = field.field_builder(100,500,display_fig=figure,results='bs')
    Fn = F_bsd - restore.dark_elaboration(field.Uniform(n),display_fig=figure)
    back = restore.bkg_est(Fn,figure)
    print(f'Extimated background maxval:\t{10**back}')
    restore.detection(Fn,back)
