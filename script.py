import numpy as np
from numpy import correlate
import matplotlib.pyplot as plt
import skysimulation.display as dpl
import skysimulation.field as fld
import skysimulation.restoration as rst
from skysimulation.field import NDArray, K, MIN_m, BETA, SEEING_SIGMA

def autocorr(vec: rst.Sequence, mode: str = 'same') -> rst.NDArray:
    return correlate(vec,vec,mode)

def plot_obj(obj0: NDArray) -> None:
    x,y = np.arange(obj0.shape[0]), np.arange(obj0.shape[1])
    x, y = np.meshgrid(x,y)
    xx, yy = x.ravel(), y.ravel()
    top = obj0[xx,yy]
    bottom = np.zeros_like(top)
    width = depth = 1
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.bar3d(xx, yy, bottom, width, depth, top, shade=True)
    from matplotlib import cm
    surf = ax.plot_surface(x,y,obj0[x,y],cmap=cm.coolwarm)#, linewidth=0, antialiased=False)
    fig.colorbar(surf)

    plt.show()    


### INITIALIZATION
# mass_seed = None
# pos_seed  = None
# bkg_seed  = None
# det_seed  = None
mass_seed = fld.M_SEED
pos_seed  = fld.POS_SEED
bkg_seed  = fld.BACK_SEED
det_seed  = fld.NOISE_SEED
# generate the field
S, (m_light, s_light), (m_dark, s_dark) = fld.field_builder(seed=(mass_seed,pos_seed), back_seed=bkg_seed, det_seed=det_seed)
# compute the scientific frame
sci_frame = m_light - m_dark
# compute the uncertainty
sigma = np.sqrt(s_light**2 + s_dark**2)
# plot it
dpl.fast_image(sci_frame,'Scientific Frame')

### RESTORING
# compute the average dark value
mean_dark = m_dark.mean()
# estimate background value
mean_bkg, sigma_bkg = rst.bkg_est(sci_frame, display_plot=False)

## Kernel Estimation
# extract objects for the kernel recovery
objs, errs, pos = rst.object_isolation(sci_frame, mean_bkg, sigma, size=5, sel_cond=True, corr_cond=False, display_fig=False)
# estimate kernel
r = rst.new_kernel_fit(objs[0], err=errs[0], display_fig=True)
