import numpy as np
import matplotlib.pyplot as plt
import skysimulation.field as field
import skysimulation.restoration as restore

if __name__ == '__main__':
    N = 100
    M = 2500
    figure = True
    back = field.BACK_PARAM
    det = field.NOISE_PARAM
    S, I = field.field_builder(N,M,back_param=back,det_param=det,display_fig=figure)

    kernel = restore.kernel_extimation()