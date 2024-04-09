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

def pipeline(*args,**kwargs) -> dict[str, fld.Any]:
    mass_seed, pos_seed, bkg_seed, det_seed = args
    ### INITIALIZATION
    # generate the field
    S, (m_light, s_light), (m_dark, s_dark) = fld.field_builder(seed=(mass_seed,pos_seed), back_seed=bkg_seed, det_seed=det_seed,**kwargs)
    # compute the scientific frame
    sci_frame = m_light - m_dark
    # compute the uncertainty
    sigma = np.sqrt(s_light**2 + s_dark**2)
    m_sigma = sigma.mean()
    s_sigma = sigma.std()
    print(f'SIGMA = {m_sigma} +- {s_sigma}')
    # plot it
    if kwargs['results']:
        dpl.fast_image(sci_frame,'Scientific Frame')

    ### RESTORING
    # compute the average dark value
    mean_dark = m_dark.mean()
    # estimate background value
    mean_bkg, sigma_bkg = rst.bkg_est(sci_frame, display_plot=False)

    ## Kernel Estimation
    # extract objects for the kernel recovery
    objs, errs, pos = rst.object_isolation(sci_frame, mean_bkg, sigma, size=5, sel_cond=True, corr_cond=False, display_fig=False,**kwargs)
    # estimate kernel
    ker_sigma, ker_Dsigma = rst.kernel_estimation(objs, errs, len(sci_frame), display_plot=False,**kwargs)

    ## R-L
    kernel = fld.Gaussian(ker_sigma).kernel()
    rec_field = rst.LR_deconvolution(sci_frame,kernel,mean_bkg,sigma,display_fig=True)

    ### STUFF
    results = {}
    results['frame']  = (sci_frame, sigma)        
    results['dark']   = (m_dark, s_dark)
    results['bkg']    = (mean_bkg, sigma_bkg)
    results['objs']   = (objs, errs, pos)
    results['seeing'] = (ker_sigma, ker_Dsigma)
    results['rl']     = rec_field

    return results        

mass_seed = fld.M_SEED
pos_seed  = fld.POS_SEED
bkg_seed  = fld.BACK_SEED
det_seed  = fld.NOISE_SEED

default_res = pipeline(mass_seed, pos_seed, bkg_seed, det_seed,results=True)

multiple_acq = False
if multiple_acq:
    iter = 3
    m_res: list[dict] = []
    mass_seed = None
    pos_seed  = None
    bkg_seed  = None
    det_seed  = None
    for i in range(iter):
        m_res += [pipeline(mass_seed, pos_seed, bkg_seed, det_seed, results=False)]
    
    seeing = np.array([ [*r['seeing']] for r in m_res])
    m_see  = seeing[:,0].mean()
    Dm_see = seeing[:,0].std()
    plt.figure()
    plt.title(f'$\\sigma = $ {m_see:.2} $\pm$ {Dm_see:.2}')
    plt.errorbar(np.arange(iter),seeing[:,0], seeing[:,1], fmt='.', linestyle='dashed')
    plt.axhline(m_see,0,1,color='red',linestyle='dotted')
    plt.axhline(m_see - Dm_see,0,1,color='orange',linestyle='dotted',alpha=0.7)
    plt.axhline(m_see + Dm_see,0,1,color='orange',linestyle='dotted',alpha=0.7)
    plt.axhline(3,0,1,color='black')
    plt.show()