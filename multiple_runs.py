import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky

DEFAULT_PARAMS = {
    'mass_seed': sky.FRAME['mass_seed'],
    'pos_seed':  sky.FRAME['pos_seed'],
    'bkg_seed':  sky.BACK['seed'],
    'det_seed':  sky.NOISE['seed']
}

def pipeline(frame_size: int = sky.FRAME['size'], star_num: int = sky.FRAME['stars'], mass_range: tuple[float,float] = sky.FRAME['mass_range'], mass_seed: float | None = None, pos_seed: float | None = None, bkg_seed: float | None = None, det_seed: float | None = None, overlap: bool = False, acq_num: int = 3, display_plots: bool = False,results:bool=True):
    ### SCIENCE FRAME
    ## Initialization
    STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(acq_num=acq_num,dim=frame_size,stnum=star_num,masses=mass_range,back_seed=bkg_seed,det_seed=det_seed,overlap=overlap,seed=(mass_seed,pos_seed),display_fig=display_plots)

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
    objs, Dobjs, objs_pos = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,num_objs=max_num_obj,cntrl=20)
    # fit a 2DGaussian profile to extraction
    ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0))
    # compute the kernel
    atm_kernel = sky.Gaussian(sigma=ker_sigma)

    ## RL Algorithm
    dec_field = sky.LR_deconvolution(sci_frame,atm_kernel,Dsci_frame,bkg_mean,bkg_sigma)

    ## Light Recover
    rec_lum, Drec_lum = sky.light_recover(dec_field,thr,bkg_mean,(ker_sigma,Dker_sigma))

    lum = np.sort(STARS.lum)[::-1]  #: initial brightness values 
    mean_lum = lum.mean()           #: mean value
    # average and compute the STD
    mean_rec, Dmean_rec = sky.mean_n_std(rec_lum)
    # print
    sky.print_measure(mean_rec,Dmean_rec,'L')
    print(f'S: {mean_lum:.2e}\t{(mean_rec-mean_lum)/Dmean_rec:.2f}')
    print(f'S: {mean_lum:.2e}\tL: {mean_rec:.2e}\t{(mean_rec-mean_lum)/mean_lum:.2%}')

    ## Plots
    try:
        # compute the magnitude between max and min data
        ratio = int(rec_lum.max()/rec_lum.min())
        bins  = int(len(rec_lum) / np.log10(ratio)) *2 if ratio != 1 else int(len(rec_lum)*2/3)
    except:
        print(rec_lum.max(),rec_lum.min())
        bins  = int(len(rec_lum)*2/3)

    if results:
        plt.figure()
        plt.hist(rec_lum,bins,histtype='step')
        plt.axvline(mean_rec,0,1,label='mean recovered brightness')
        plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
        plt.axvline(mean_lum,0,1,color='red',label='mean source brightness')
        plt.legend()
        plt.show()
    return mean_lum, mean_rec

if __name__ == '__main__':
    
    ## Constants
    DISPLAY_PLOTS = False

    ## Overlap
    # _ = pipeline(**DEFAULT_PARAMS,overlap=True)

    ## Sparsely and Overpopulated
    # 
    _ = pipeline(star_num=30,**DEFAULT_PARAMS,overlap=True)
    _ = pipeline(star_num=500,**DEFAULT_PARAMS,overlap=True)

    ## Random no overlap
    iterations = 10
    ratio = []
    for _ in iterations:
        lum, rec = pipeline(results=False)
        ratio += [lum/rec]
    plt.figure()
    plt.plot(ratio,'.--')
    plt.axhline(1,0,1,color='black')
    plt.show()