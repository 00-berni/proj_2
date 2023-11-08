import numpy as np
import skysimulation.field as field
from skysimulation.display import fast_image

if __name__ == '__main__':
    figure = True
    # F_bsd, F_b, F_bs = field.field_builder(display_fig=figure,results='bs')
    F, S = field.initialize(dim=10,sdim=1,display_fig=True)
    Fs = field.atm_seeing(F,display_fig=True)