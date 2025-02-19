import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky

DEFAULT_PARAMS = {
    'mass_seed': sky.FRAME['mass_seed'],
    'pos_seed':  sky.FRAME['pos_seed'],
    'bkg_seed':  sky.BACK['seed'],
    'det_seed':  sky.NOISE['seed']
}
BINNING = 63
FONTSIZE = 18
ITERATIONS = 2000

### MONTE CARLO REALIZATIONS
def generate_sample(**initargs) -> sky.NDArray:
    initargs['p_seed'] = None
    _, S = sky.initialize(**initargs)
    distances = sky.dist_corr(S.pos)
    return distances


def pipeline(frame_size: int = sky.FRAME['size'], star_num: int = sky.FRAME['stars'], mass_range: tuple[float,float] = sky.FRAME['mass_range'], mass_seed: float | None = None, pos_seed: float | None = None, bkg_seed: float | None = None, det_seed: float | None = None, bkg_param: tuple = sky.field.BACK_PARAM, overlap: bool = False, acq_num: int = 6, montecarlo: bool = True, iter: int = ITERATIONS, display_plots: bool = False,results:bool=True) -> tuple[float,float, None] | tuple[float, float, sky.NDArray]:
    ### SCIENCE FRAME
    ## Initialization
    STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(acq_num=acq_num,dim=frame_size,stnum=star_num,masses=mass_range,back_param=bkg_param,back_seed=bkg_seed,det_seed=det_seed,overlap=overlap,seed=(mass_seed,pos_seed),display_fig=display_plots,results=results)

    ## Science Frame
    sci_frame = master_light - master_dark 
    Dsci_frame = np.sqrt(Dmaster_light**2 + Dmaster_dark**2)

    ### RESTORATION
    ## Background Estimation
    (bkg_mean, _), bkg_sigma = sky.bkg_est(sci_frame,display_plot=display_plots)

    ## Kernel Estimation
    thr = bkg_mean + bkg_sigma  #: the threshold for searching algorithm
    max_num_obj = 10            #: number of objects at the most 
    # extract objects
    objs, Dobjs, objs_pos = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,num_objs=max_num_obj,cntrl=20,display_fig=results)
    # fit a 2DGaussian profile to extraction
    ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0),results=results)
    # compute the kernel
    atm_kernel = sky.Gaussian(sigma=ker_sigma)

    ## RL Algorithm
    dec_field = sky.LR_deconvolution(sci_frame,atm_kernel,Dsci_frame,bkg_mean,bkg_sigma,results=results)

    ## Light Recover
    rec_res = {} if montecarlo else None
    rec_lum, Drec_lum = sky.light_recover(dec_field,thr,bkg_mean,(ker_sigma,Dker_sigma),extraction=rec_res, results=results,display_fig=results)

    lum = np.sort(STARS.lum)[::-1]  #: initial brightness values 
    mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()          #: mean value
    # average and compute the STD
    mean_rec, Dmean_rec = sky.mean_n_std(rec_lum)
    # print
    print('SUM COMPARE',STARS.lum[::-1][:4])
    sky.print_measure(mean_rec,Dmean_rec,'L')
    print(f'S: {mean_lum:.2e}\t{(mean_rec-mean_lum)/Dmean_rec:.2f}')
    print(f'S: {mean_lum:.2e}\tL: {mean_rec:.2e}\t{(mean_rec-mean_lum)/mean_lum:.2%}')

    ## Plots

    if results:
        plt.figure()
        plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
        plt.hist(rec_lum,BINNING,histtype='step')
        plt.axvline(mean_rec,0,1,label='mean recovered brightness',linestyle='dotted')
        # plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
        plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dotted')
        plt.legend(fontsize=FONTSIZE)
        plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
        plt.ylabel('counts',fontsize=FONTSIZE)
        plt.figure()
        plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
        bins = np.linspace(min(STARS.lum.min(),rec_lum.min()),max(STARS.lum.max(),rec_lum.max()),BINNING)
        plt.hist(STARS.lum,bins,density=True,color='red',histtype='step')
        plt.hist(rec_lum,bins,density=True,color='blue',histtype='step')
        plt.axvline(mean_rec,0,1,label='mean recovered brightness',linestyle='dotted')
        # plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
        plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dotted')
        plt.legend(fontsize=FONTSIZE)
        plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
        plt.ylabel('counts',fontsize=FONTSIZE)
        plt.show()
        
    if montecarlo:
        obj_pos = rec_res['pos']
        fig, ax = plt.subplots(1,1)
        sky.field_image(fig,ax,dec_field)
        ax.plot(*obj_pos[::-1],'.',color='blue')
        ax.plot(STARS.pos[1],STARS.pos[0],'x',color='red')
        plt.show()
        cal_dist = lambda p1,p2 : np.sqrt(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
        counts = np.array([len(np.where(cal_dist(sp,obj_pos) <= ker_sigma)[0]) for sp in STARS.pos])
        print('COUNTS',len(counts[counts==1]))
        print('COUNTS STARS',len(STARS.lum[STARS.lum>bkg_mean]))
        rec_dist = sky.dist_corr(obj_pos)
        if results:
            gen_dists = np.array([generate_sample(dim=frame_size,sdim=star_num,masses=mass_range,overlap=overlap,m_seed=mass_seed) for _ in range(iter)])
            mean_dist = np.mean(gen_dists)
            std_dist  = np.std(np.mean(gen_dists,axis=1))
            print(np.mean(gen_dists,axis=1).shape, mean_dist, std_dist)
            gen_dist = gen_dists.flatten()
            plt.figure()
            _, bins, _ = plt.hist(gen_dist,BINNING,density=True,color='red',histtype='step')
            plt.hist(rec_dist,bins,density=True,color='blue',histtype='step')
            plt.axvline(mean_dist,0,1,color='red',linestyle='dotted',label=f'gen mean = {mean_dist:.2f} +/- {std_dist:.2f}')
            plt.axvspan(mean_dist-std_dist,mean_dist+std_dist,facecolor='red',alpha=0.3)            
            plt.axvline(rec_dist.mean(),0,1,color='blue',linestyle='dotted',label=f'rec mean = {rec_dist.mean():.2f}')  
            plt.legend(fontsize=FONTSIZE)
            plt.title('Distribution of distances',fontsize=FONTSIZE+2)
            plt.show()          
        rec_res = rec_dist
    return mean_lum, mean_rec, rec_res

if __name__ == '__main__':
    
    ## Constants
    DISPLAY_PLOTS = False

    params = DEFAULT_PARAMS.copy()

    ### DEFAULT WITH OVERLAP 
    # _ = pipeline(**DEFAULT_PARAMS,overlap=True)

    # ### SPARSELY POPULATED 
    # params['mass_seed'] += 3
    # params['pos_seed']  += 3
    # _ = pipeline(star_num=50,**params,overlap=True,iter=3000)
    
    # ### OVER-POPULATED 
    # params['mass_seed'] += 3
    # params['pos_seed']  += 3
    # _ = pipeline(star_num=500,**params,overlap=True, iter=1000)

    ### DIFFERNT BACKGROUNDS
    bkg_mean = [sky.BACK_MEAN/2,sky.BACK_MEAN*2,sky.BACK_MEAN*3]
    lums = np.array([])
    recs = np.array([])
    dist = np.array([])
    for bkg in bkg_mean:
        print('\n\n~ ~ BACKGROUND ~ ~',bkg/sky.BACK_MEAN)
        check = True
        fails = 0
        while check:
            try:
                l,r,d = pipeline(**DEFAULT_PARAMS,bkg_param=('Gaussian',(bkg,bkg*20e-2)))
                check = False
            except:
                fails += 1
                print('FAILED:',fails)
                raise
        lums = np.append(lums,[l])
        recs = np.append(recs,[r])
        dist = np.append(dist,[np.mean(d)])
    ratio = (recs-lums)/lums
    print()
    print('RATIO',ratio*100)
    print('DISTS',dist)
    