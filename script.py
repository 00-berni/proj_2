import numpy as np
from numpy import correlate
import matplotlib.pyplot as plt
import skysimulation.display as dpl
import skysimulation.stuff as stf
import skysimulation.field as fld
import skysimulation.restoration as rst
from skysimulation.stuff import NDArray, sqr_mask




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
    """_summary_

    Returns
    -------
    results : dict[str, Any]
        - `results['frame']  =  (sci_frame: NDArray, sigma: NDArray)`
        - `results['dark']   =  (m_dark: NDArray, s_dark: NDArray)`
        - `results['bkg']    =  (m_bkg: float, sigma_bkg: float)`
        - `results['objs']   =  (objs: list[NDArray], errs: list[NDArray], pos: NDArray)`
        - `results['seeing'] =  (ker_sigma: float, ker_Dsigma: float)`
        - `results['rl']     =  rec_field: NDArray`
    """
    ### STUFF
    mass_seed, pos_seed, bkg_seed, det_seed = args
    if 'method' in kwargs: 
        method = kwargs['method'] if kwargs['method'] is not None else 'all'
        kwargs.pop('method')
    else:
        method = 'all'
    if 'disp_res' not in kwargs:
        kwargs['disp_res'] = True
    disp_res = kwargs['disp_res']
    kwargs.pop('disp_res')
    results = {}
    last_sen = []

    
    ### INITIALIZATION
    # generate the field
    S, (m_light, s_light), (m_dark, s_dark) = fld.field_builder(seed=(mass_seed,pos_seed), back_seed=bkg_seed, det_seed=det_seed,**kwargs)
    last_sen += [f'L0: {S.lum.max()}\nL0.5: {m_light.max()*np.sqrt(18*np.pi)}']
    # compute the scientific frame
    sci_frame = m_light - m_dark
    last_sen += [f'L1: {sci_frame.max()*np.sqrt(18*np.pi)}']
    # compute the unceresultsrtainty
    sigma = np.sqrt(s_light**2 + s_dark**2)
    m_sigma = sigma.mean()
    s_sigma = sigma.std()
    print(f'SIGMA = {m_sigma} +- {s_sigma}')
    # plot it
    if disp_res:
        dpl.fast_image(sci_frame,'Scientific Frame',norm='log')

    print('!!CHECK!!\t',len(np.where(S.lum > fld.BACK_MEAN*fld.K)[0]))
    results['stars']  = S
    results['frame']  = (sci_frame, sigma)        
    results['dark']   = (m_dark, s_dark)

    ### RESTORING
    if method in ['all','rl','kernel','obj','bkg']:
        max_size = 5
        # compute the average dark value
        mean_dark = m_dark.mean()
        # estimate background value
        m_bkg, sigma_bkg = rst.bkg_est(sci_frame, display_plot=True)
        mean_bkg, Dmean_bkg = m_bkg
        print('Back',fld.BACK_MEAN*fld.K, fld.BACK_SIGMA*fld.K)
        results['bkg'] = (m_bkg, sigma_bkg)
   
    ## Kernel Estimation
    if method in ['all','rl','kernel','obj']:
        # extract objects for the kernel recovery
        objs, errs, pos = rst.searching(sci_frame, mean_bkg+sigma_bkg, mean_bkg, errs=sigma, max_size=max_size, min_dist=2, num_objs=10,cntrl_mode='bright', display_fig=False,**kwargs)
        results['objs'] = (objs, errs, pos)


    if method in ['all','rl','kernel']:
        # estimate kernel
        ker_sigma, ker_Dsigma = rst.kernel_estimation(objs, errs, m_bkg, display_plot=False,**kwargs)
        results['seeing'] = (ker_sigma, ker_Dsigma)

    ## R-L
    if method in ['all', 'rl']:
        # compute the estimated kernel
        kernel = stf.Gaussian(ker_sigma)
        rec_field = rst.LR_deconvolution(sci_frame,kernel,sigma, mean_bkg, sigma_bkg, display_fig=True)
        if disp_res: 
            dpl.fast_image(rec_field - sci_frame - mean_bkg,'Remove before and background')
        results['rl'] = rec_field
        flat_field = rec_field.flatten()
        if disp_res:
            plt.figure()
            plt.hist(flat_field,100)
            plt.yscale('log')
            fig, ax = plt.subplots(1,1)
            dim = len(rec_field)
            dpl.field_image(fig, ax, rec_field)
            mask0 = sqr_mask(ker_sigma, dim)
            ax.plot(mask0[:,0], mask0[:,1], color='blue')
            plt.show()

        last_sen += [f'L2: {rec_field.max()*np.sqrt(18*np.pi)}']
        art_bkg = stf.Gaussian(sigma_bkg, mean_bkg).field(rec_field.shape)
        if disp_res:
            dpl.fast_image(rec_field - art_bkg, 'Removed background')

    ## Light Recovery
    if method == 'all':
        objs, err, pos = rst.searching(sci_frame, mean_bkg+sigma_bkg, mean_bkg, errs=None, max_size=5, **kwargs)
        maxvalues = np.array([o.max() for o in objs])
        maxerrs   = np.array([err[i][stf.peak_pos(objs[i])] for i in range(len(objs))])
        rec_lum  = maxvalues*np.sqrt(2*np.pi*ker_sigma**2)
        Drec_lum = rec_lum * np.sqrt((maxerrs/maxvalues)**2 + (ker_Dsigma/ker_sigma)**2)
        results['reclum'] = (rec_lum, Drec_lum) 
        plt.figure()
        plt.hist(rec_lum, len(rec_lum)//2, histtype='step',color='red',label='Max Val')
        plt.show()
        
    results['last_sen'] = last_sen
    return results        

### MAIN ###
if __name__ == '__main__':
    mass_seed = fld.M_SEED
    pos_seed  = fld.POS_SEED
    bkg_seed  = fld.BACK_SEED
    det_seed  = fld.NOISE_SEED
    # mass_seed = None
    # pos_seed  = None
    # bkg_seed  = None
    # det_seed  = None
    method = 'all'
    # method = None
    default_res = pipeline(mass_seed, pos_seed, bkg_seed, det_seed, method=method, disp_res=True)
    exit()
    mean_bkg = default_res['bkg'][0][0]
    field, sigma = default_res['frame']    
    star = default_res['stars']
    lum = star.lum
    objs = default_res['objs'][0]
    pos = default_res['objs'][-1]
    rec_field = default_res['rl']
    ker_sigma, ker_Dsigma = default_res['seeing']
    objs, err, pos = rst.searching(rec_field, mean_bkg, sigma, max_size=5, cntrl=None, cntrl_sel='bright', debug_plots=False)
    # objs, err, pos = rst.searching(rec_field, mean_bkg*105e-100, sigma, max_size=5, cntrl=None, cntrl_sel='bright', debug_plots=False)
    m_dark = default_res['dark'][0]
    maxvalues  = np.array([ rec_field[x,y] for x, y in zip(*pos)])
    val_obj0 = maxvalues.max()*np.sqrt(2*np.pi*ker_sigma**2)
    err_obj0 = rec_field[pos[1,0],pos[0,0]]
    err_obj0 = np.sqrt((err_obj0/maxvalues.max())**2 + (ker_Dsigma/ker_sigma)**2) * val_obj0
    print('sigma',ker_sigma)
    print('\nLuminosities:')
    for sen in default_res['last_sen']:
        print(sen)
    print(f'L3: {maxvalues.max()*np.sqrt(2*np.pi*ker_sigma**2)} +/- {err_obj0}')
    print(f'L4: {objs[0].max()*np.sqrt(2*np.pi*ker_sigma**2)}')
    print('VAL',maxvalues.max()*np.sqrt(2*np.pi*ker_sigma**2)/lum.max(),err_obj0/lum.max())
    print('VAL',(maxvalues.max()+m_dark.mean())*np.sqrt(2*np.pi*ker_sigma**2)/lum.max())

    """
        IMF = M^-a
        L = M^b
        IMF = 1/b * L^{(1-a-b)/b}
        E[L] = int L * IMF = int L^{1-a/b} = L^{2-a/b}/(2-a/b) =
        = b/(2b-a) * (LM^{(2b-a)/b}-Lm^{(2b-a)/b})   
    """

    # from scipy.integrate import trapezoid
    # maxvalues = np.array([ trapezoid(trapezoid(obj)) for obj in objs])
    plt.figure()
    binnn = lambda arr : 10**(np.linspace(np.log10(arr).min(),np.log10(arr).max(),10))
    plt.hist(maxvalues,30, histtype='step',color='red',label='Max Val')
    plt.hist((maxvalues+m_dark.mean())*np.sqrt(2*np.pi*ker_sigma**2),30, histtype='step',color='green',label='prod')
    # plt.hist(maxvalues*2*np.pi*ker_sigma**2,30, histtype='step',color='green',label='prod')
    plt.hist(lum,60, histtype='step', label='Lum',color='blue')
    # plt.xscale('log')
    plt.axvline(np.mean(maxvalues),0,1,linestyle='dashed',color='red')
    plt.axvline(np.mean(lum),0,1,linestyle='dashed',color='blue')
    plt.axvline(mean_bkg,0,1,color='violet')
    plt.yscale('log')
    plt.legend()
    plt.show()
    print(f'\n\n------\n\nFOUND:\t{len(objs)}\nOBSER:\t{len(lum[lum>mean_bkg])}')
    
    ## Expectation Values
    def power_law(x, n, k):
        return k*x**n
    
    maxvalues *= np.sqrt(2*np.pi*ker_sigma)
    cnts, bins = np.histogram(maxvalues,40)
    xdata = (bins[1:]+bins[:-1])/2
    xerr  = (bins[1:]-bins[:-1])/2
    fit = rst.FuncFit(xdata=xdata,xerr=xerr,ydata=cnts)
    fit.pipeline(power_law,[-1,cnts.max()],names=['-a/b','K'])
    (n,k), (Dn,Dk) = fit.results()

    plt.figure()
    xx = np.linspace(xdata.min(),xdata.max(),300)
    plt.subplot(2,1,1)
    plt.errorbar(xdata,cnts,xerr=xerr,fmt='.')
    plt.plot(xx,power_law(xx,*fit.fit_par))
    plt.plot(xx,k*xx**(-4/3))
    plt.subplot(2,1,2)
    plt.errorbar(xdata,cnts-power_law(xdata,n,k),abs(n*power_law(xdata,n,k)*xerr/xdata),fmt='.')
    plt.axhline(0,0,1,color='black')
    
    plt.figure()
    cnts,_ = np.histogram(lum,60)
    plt.hist(lum,60)
    plt.plot(lum,cnts.max()*lum**(-4/3))
    plt.show()
    # a = -2
    # b  = a/n
    # Db = b * Dn/n
    # exp_val0 = 3/4 * (lum.max()**(3/4) - lum.min()**(3/4))
    # exp_val  = b/(b-1)/2 * (maxvalues.max()**(2*(b-1)/b) - maxvalues.min()**(2*(b-1)/b))


    # print('EXP',exp_val,exp_val0)

    # plt.figure()
    # plt.hist(maxvalues,30, histtype='step',color='red',label='Max Val')
    # plt.hist(lum,60, histtype='step', label='Lum',color='blue')
    # plt.axvline(exp_val,0,1,color='red',linestyle='--',label=f'{exp_val}')
    # plt.axvline(exp_val0,0,1,color='blue',linestyle='--',label=f'{exp_val0}')
    plt.show()





    # multiple_acq = False
    # if multiple_acq:
    #     iter = 5
    #     m_res: list[dict] = []
    #     mass_seed = None
    #     pos_seed  = None
    #     bkg_seed  = None
    #     det_seed  = None
    #     for i in range(iter):
    #         m_res += [pipeline(mass_seed, pos_seed, bkg_seed, det_seed, method=method, results=True)]
        
    #     # background = np.array([[*r['bkg']] for r in m_res])
    #     m_bkg = np.array([[*r['bkg'][0]] for r in m_res])
    #     s_bkg = np.array([ r['bkg'][1] for r in m_res])
    #     # m_bkg = background[:,0]
    #     # s_bkg = background[:,1]
    #     plt.figure()
    #     plt.suptitle('Background')
    #     plt.subplot(1,2,1)
    #     m,s = rst.mean_n_std(m_bkg[:,0])
    #     plt.title(f'mean = {m:.4} $\\pm$ {s:.2}')
    #     plt.errorbar(np.arange(iter),m_bkg[:,0],m_bkg[:,1],fmt='.', linestyle='dashed')
    #     plt.axhline(m,0,1,color='red',linestyle='dotted')
    #     plt.axhline(m - s,0,1,color='orange',linestyle='dotted',alpha=0.7)
    #     plt.axhline(m + s,0,1,color='orange',linestyle='dotted',alpha=0.7)
    #     plt.axhline(fld.BACK_MEAN*fld.K,0,1,color='black')
    #     plt.subplot(1,2,2)
    #     m,s = rst.mean_n_std(s_bkg)
    #     plt.title(f'$\\sigma = $ {m:.4} $\\pm$ {s:.2}')
    #     plt.plot(np.arange(iter),s_bkg,'.--')
    #     plt.axhline(m,0,1,color='red',linestyle='dotted')
    #     plt.axhline(m - s,0,1,color='orange',linestyle='dotted',alpha=0.7)
    #     plt.axhline(m + s,0,1,color='orange',linestyle='dotted',alpha=0.7)
    #     plt.axhline(fld.BACK_SIGMA*fld.K,0,1,color='black')
    #     plt.show()    
        
    #     seeing = np.array([ [*r['seeing']] for r in m_res])
    #     m_see, Dm_see = rst.mean_n_std(seeing[:,0])
    #     plt.figure()
    #     plt.title(f'$\\sigma = $ {m_see:.4} $\\pm$ {Dm_see:.2}')
    #     plt.errorbar(np.arange(iter),seeing[:,0], seeing[:,1], fmt='.', linestyle='dashed')
    #     plt.axhline(m_see,0,1,color='red',linestyle='dotted')
    #     plt.axhline(m_see - Dm_see,0,1,color='orange',linestyle='dotted',alpha=0.7)
    #     plt.axhline(m_see + Dm_see,0,1,color='orange',linestyle='dotted',alpha=0.7)
    #     plt.axhline(3,0,1,color='black')
    #     plt.show()