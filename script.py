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
        kwargs = {key: kwargs[key] for key in kwargs.keys()-{'method'} }
    else:
        method = 'all'
    results = {}
    
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
        objs, errs, pos = rst.searching(sci_frame, mean_bkg, sigma, max_size=max_size, min_dist=2, display_fig=False,**kwargs)
        results['objs'] = (objs, errs, pos)


    if method in ['all','rl','kernel']:
        # estimate kernel
        ker_sigma, ker_Dsigma = rst.kernel_estimation(objs[:6], errs[:6], m_bkg, display_plot=False,**kwargs)
        results['seeing'] = (ker_sigma, ker_Dsigma)

    ## R-L
    if method in ['all', 'rl']:
        # compute the estimated kernel
        kernel = stf.Gaussian(ker_sigma)
        rec_field = rst.LR_deconvolution(sci_frame,kernel,sigma, mean_bkg, sigma_bkg, display_fig=True)
        dpl.fast_image(rec_field - sci_frame - mean_bkg,'Remove before and background')
        results['rl'] = rec_field
        
        fig, ax = plt.subplots(1,1)
        dim = len(rec_field)
        dpl.field_image(fig, ax, rec_field)
        mask0 = sqr_mask(ker_sigma, dim)
        ax.plot(mask0[:,0], mask0[:,1], color='blue')
        plt.show()

        art_bkg = stf.Gaussian(sigma_bkg, mean_bkg).field(rec_field.shape)
        dpl.fast_image(rec_field - art_bkg, 'Removed background')

    ## Light Recovery
    if method == 'all':
        _ = rst.searching(rec_field, mean_bkg, None, max_size=max_size, debug_plots=True, **kwargs)
        # det_stars = np.where(S.lum > mean_bkg)[0]
        # det_pos = np.array(S.pos)[:,det_stars]
        # dist = lambda p1, p2 : np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        # objs, errs, pos = rst.find_objects(sci_frame, rec_field, mean_bkg, sigma, max_size, results=False, display_fig=True)
        # ok_pos = [ [*np.where(dist(det_pos[:,i], pos) < 2)] for i in range(len(det_stars)) ]
        # print('POS')
        # print(len(ok_pos))
        # print(len(ok_pos[0]))
        # print(ok_pos)
        # if len(ok_pos[0]) != 0:
        #     ok_obj = np.array([ pos[:,p[0]] for p in ok_pos if 0 not in p[0].shape ])
        #     print(ok_obj.shape)
        #     print(ok_obj)
        # fig, ax = plt.subplots(1,1)
        # ax.set_title('Detected Objects vs Detectable Objects')
        # dpl.field_image(fig,ax,rec_field)
        # ax.plot(pos[1],pos[0],'.',color='blue',label='chosen objects')
        # ax.plot(det_pos[1],det_pos[0],'x',color='violet',label='detectable stars')
        # if len(ok_pos[0]) != 0:
        #     for i in range(ok_obj.shape[-1]):
        #         ax.plot(ok_obj[:,1,i],ok_obj[:,0,i],'+',color='red',label='good')
        # ax.legend()
        # plt.show()
        # print(f'OBSERVATED::\t{len(objs)}')
        # print(f'OBSERVABLE::\t{len(det_stars)}')
        # print(f'WRONG::\t{len(objs)-len(ok_obj)}')
    # lums = np.sort(S.lum)[::-1]
    # l0 = []
    # l1 = []
    # from scipy.integrate import trapezoid
    # for (obj, err, lum) in zip(objs[:3],errs[:3],lums[:3]):
    #     print(len(obj))
    #     r_obj = rst.LR_deconvolution(obj,kernel,err,mean_bkg,sigma_bkg, display_fig=False)
    #     # err = np.sqrt(err**2 + err.var())
    #     (_,s,xmax,ymax), _ = rst.new_kernel_fit(r_obj,err=err,display_fig=False)
    #     xdim, ydim = r_obj.shape
    #     x_edge = slice(max(0, int(xmax - s)), min(xdim, int(xmax + s) +1))
    #     y_edge = slice(max(0, int(ymax - s)), min(ydim, int(ymax + s) +1))
    #     cobj = r_obj[x_edge,y_edge]
    #     l0 += [cobj.sum()-mean_bkg]
    #     print(f'CHECK::\t{lum}\t{cobj.sum()}\t{cobj.sum()-mean_bkg}')
    #     s += 1
    #     x_edge = slice(max(0, int(xmax - s)), min(xdim, int(xmax + s) +1))
    #     y_edge = slice(max(0, int(ymax - s)), min(ydim, int(ymax + s) +1))
    #     cobj = r_obj[x_edge,y_edge]
    #     l1 += [cobj.sum()-mean_bkg]
    #     # xmax, ymax = rst.peak_pos(r_obj)
    #     # r_objmax = r_obj[xmax,ymax]
    #     # pos = np.unravel_index(abs(r_obj - r_objmax/2).argmin(), r_obj.shape)
    #     # w = max( abs(pos[0] - xmax), abs(pos[1] - ymax )) +1
    #     # dpl.fast_image(r_obj - obj)
    #     # dpl.fast_image(r_obj[x_edge,y_edge],f'({xmax},{ymax}),{pos}, {w}')
    #     # fig, (ax1,ax2) = plt.subplots(1,2)
    #     # rsum = r_obj[x_edge,y_edge].sum() - mean_bkg
    #     # fig.suptitle(f'star = {lum:.3} ; r_max = {r_obj.max() - mean_bkg:.3} ; sum = {rsum:.3} +- {sigma.std():.3}\nover = {(rsum - lum)/rsum*100:.2f} %')
    #     # dpl.field_image(fig,ax1,obj)
    #     # dpl.field_image(fig,ax2,r_obj)
    #     # plt.show()
    # plt.figure()
    # plt.bar(np.arange(3),lums[:3],0.2,align='center',color='blue',label='lux')
    # plt.bar(np.arange(3)-0.2,l0,0.2,align='center',color='red',label='s')
    # plt.bar(np.arange(3)+0.2,l1,0.2,align='center',color='orange',label='s+1')
    # plt.legend()
    # plt.show()


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
    method = 'rl'
    # method = None
    default_res = pipeline(mass_seed, pos_seed, bkg_seed, det_seed, method=method, results=True)
    mean_bkg = default_res['bkg'][0][0]
    field, sigma = default_res['frame']    
    star = default_res['stars']
    lum = star.lum
    objs = default_res['objs'][0]
    pos = default_res['objs'][-1]
    rec_field = default_res['rl']
    objs, err, pos = rst.searching(rec_field, mean_bkg*105e-100, sigma, max_size=5, cntrl=None, cntrl_sel='bright', debug_plots=False)


    maxvalues = np.array([ field[x,y] for x, y in zip(*pos)])
    # from scipy.integrate import trapezoid
    # maxvalues = np.array([ trapezoid(trapezoid(obj)) for obj in objs])
    plt.figure()
    binnn = lambda arr : 10**(np.linspace(np.log10(arr).min(),np.log10(arr).max(),10))
    plt.hist(maxvalues,binnn(maxvalues), histtype='step',color='red',label='Max Val')
    plt.hist(lum,binnn(lum), histtype='step', label='Lum',color='blue')
    plt.xscale('log')
    plt.axvline(np.mean(maxvalues),0,1,linestyle='dashed',color='red')
    plt.axvline(np.mean(lum),0,1,linestyle='dashed',color='blue')
    plt.legend()
    plt.show()
    print(f'\n\n------\n\nFOUND:\t{len(objs)}\nOBSER:\t{len(lum[lum>mean_bkg])}')
    

    multiple_acq = False
    if multiple_acq:
        iter = 5
        m_res: list[dict] = []
        mass_seed = None
        pos_seed  = None
        bkg_seed  = None
        det_seed  = None
        for i in range(iter):
            m_res += [pipeline(mass_seed, pos_seed, bkg_seed, det_seed, method=method, results=True)]
        
        # background = np.array([[*r['bkg']] for r in m_res])
        m_bkg = np.array([[*r['bkg'][0]] for r in m_res])
        s_bkg = np.array([ r['bkg'][1] for r in m_res])
        # m_bkg = background[:,0]
        # s_bkg = background[:,1]
        plt.figure()
        plt.suptitle('Background')
        plt.subplot(1,2,1)
        m,s = rst.mean_n_std(m_bkg[:,0])
        plt.title(f'mean = {m:.4} $\\pm$ {s:.2}')
        plt.errorbar(np.arange(iter),m_bkg[:,0],m_bkg[:,1],fmt='.', linestyle='dashed')
        plt.axhline(m,0,1,color='red',linestyle='dotted')
        plt.axhline(m - s,0,1,color='orange',linestyle='dotted',alpha=0.7)
        plt.axhline(m + s,0,1,color='orange',linestyle='dotted',alpha=0.7)
        plt.axhline(fld.BACK_MEAN*fld.K,0,1,color='black')
        plt.subplot(1,2,2)
        m,s = rst.mean_n_std(s_bkg)
        plt.title(f'$\\sigma = $ {m:.4} $\\pm$ {s:.2}')
        plt.plot(np.arange(iter),s_bkg,'.--')
        plt.axhline(m,0,1,color='red',linestyle='dotted')
        plt.axhline(m - s,0,1,color='orange',linestyle='dotted',alpha=0.7)
        plt.axhline(m + s,0,1,color='orange',linestyle='dotted',alpha=0.7)
        plt.axhline(fld.BACK_SIGMA*fld.K,0,1,color='black')
        plt.show()    
        
        seeing = np.array([ [*r['seeing']] for r in m_res])
        m_see, Dm_see = rst.mean_n_std(seeing[:,0])
        plt.figure()
        plt.title(f'$\\sigma = $ {m_see:.4} $\\pm$ {Dm_see:.2}')
        plt.errorbar(np.arange(iter),seeing[:,0], seeing[:,1], fmt='.', linestyle='dashed')
        plt.axhline(m_see,0,1,color='red',linestyle='dotted')
        plt.axhline(m_see - Dm_see,0,1,color='orange',linestyle='dotted',alpha=0.7)
        plt.axhline(m_see + Dm_see,0,1,color='orange',linestyle='dotted',alpha=0.7)
        plt.axhline(3,0,1,color='black')
        plt.show()