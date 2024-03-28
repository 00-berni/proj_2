import numpy as np
from numpy import correlate
import matplotlib.pyplot as plt
import skysimulation.display as dpl
import skysimulation.field as fld
import skysimulation.restoration as rst
from skysimulation.field import NDArray, K, MIN_m, BETA, SEEING_SIGMA

def autocorr(vec: rst.Sequence, mode: str = 'same') -> rst.NDArray:
    return correlate(vec,vec,mode)

if __name__ == '__main__':
    ### INITIALIZATION
    # mass_seed = None
    # pos_seed  = None
    # bkg_seed  = None
    # det_seed  = None
    mass_seed = fld.M_SEED
    pos_seed  = fld.POS_SEED
    bkg_seed  = fld.BACK_SEED
    det_seed  = fld.NOISE_SEED
    S, (m_light, s_light), (m_dark, s_dark) = fld.field_builder(seed=(mass_seed,pos_seed), back_seed=bkg_seed, det_seed=det_seed)
    sci_frame = m_light - m_dark
    sigma = np.sqrt(s_light**2 + s_dark**2)
    dpl.fast_image(sci_frame,'Scientific Frame')

    ### RESTORING
    # compute the average dark value
    mean_dark = m_dark.mean()
    # estimate background value
    mean_bkg, sigma_bkg = rst.bkg_est(sci_frame, display_plot=True)
    # extract object for the fit
    obj, pos = rst.object_isolation(sci_frame,mean_bkg, sel_cond=True, corr_cond=False,display_fig=True)


