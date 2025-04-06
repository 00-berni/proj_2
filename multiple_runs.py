import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky
from skysimulation import log_update

DEFAULT_PARAMS = {
    'mass_seed': sky.FRAME['mass_seed'],
    'pos_seed':  sky.FRAME['pos_seed'],
    'bkg_seed':  sky.BACK['seed'],
    'det_seed':  sky.NOISE['seed']
}
BINNING = 20
FONTSIZE = 18
ITERATIONS = 2000
BKG_MEAN = sky.BACK_MEAN

### MONTE CARLO REALIZATIONS
def generate_sample(selection: None | sky.ArrayLike = None,**initargs) -> sky.NDArray:
    _, S = sky.initialize(**initargs,p_seed=None)
    pos = np.asarray(S.pos)
    if selection is not None:
        pos = pos[:,S.lum > selection]
    distances = sky.dist_corr(pos)
    return distances

def science_frame(frame_size: int = sky.FRAME['size'], star_num: int = sky.FRAME['stars'], mass_range: tuple[float,float] = sky.FRAME['mass_range'], mass_seed: float | None = None, pos_seed: float | None = None, bkg_seed: float | None = None, det_seed: float | None = None, bkg_param: tuple = sky.field.BACK_PARAM, overlap: bool = True, acq_num: int = 6,display_plots: bool = False, results:bool=True) -> tuple[sky.Star, tuple[sky.NDArray,sky.NDArray]]:
    ## Initialization
    STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(acq_num=acq_num,dim=frame_size,stnum=star_num,masses=mass_range,back_param=bkg_param,back_seed=bkg_seed,det_seed=det_seed,overlap=overlap,seed=(mass_seed,pos_seed),results=results,display_fig=display_plots)
    if results:
        STARS.plot_info()
        plt.show()
    print('CHECK',sky.BACK_MEAN*sky.K,len(STARS.lum[STARS.lum<sky.BACK_MEAN*sky.K]),STARS.lum.max(),STARS.lum[:4])
    ## Science Frame
    sci_frame = master_light - master_dark 
    Dsci_frame = np.sqrt(Dmaster_light**2 + Dmaster_dark**2)
    if results:
        sky.fast_image(sci_frame,'Science Frame')
        plt.figure()
        plt.hist(Dsci_frame.flatten(),71)
        plt.show()
    return STARS, (sci_frame, Dsci_frame)

def restoration(sci_frame: sky.NDArray, Dsci_frame: sky.NDArray, stars: sky.Star, display_plots: bool = False, results:bool=True, verbose_display: bool = False, checks:bool=True,stop_ctrl:bool = False) -> tuple[tuple[sky.NDArray,sky.NDArray], dict]:
    params = {}
    ## Background Estimation
    (bkg_mean, _), bkg_sigma = sky.bkg_est(sci_frame,display_plot=results)
    params['bkg'] = {'mean': bkg_mean, 'sigma': bkg_sigma}

    ## Kernel Estimation
    thr = bkg_mean + bkg_sigma  #: the threshold for searching algorithm
    max_num_obj = 10            #: number of objects at the most 
    obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
    # extract objects
    objs, Dobjs, _ = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,ker_sigma=None,num_objs=max_num_obj,cntrl=20,log=True,obj_params=obj_param,display_fig=results,debug_plots=verbose_display)
    print(obj_param)
    # fit a 2DGaussian profile to extraction
    ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0),obj_param=obj_param,results=results,display_plot=verbose_display,title='title-only')
    # compute the kernel
    atm_kernel = sky.Gaussian(sigma=ker_sigma)
    params['kernel'] = {'sigma': ker_sigma, 'Dsigma': Dker_sigma, 'kernel': atm_kernel}
    
    if stop_ctrl: return None
    
    ## RL Algorithm
    dec_field = sky.LR_deconvolution(sci_frame,atm_kernel,Dsci_frame,bkg_mean,bkg_sigma,results=results)

    if results:
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.set_title('Before deconvolution',fontsize=FONTSIZE+2)
        sky.field_image(fig,ax1,sci_frame)
        ax2.set_title('After deconvolution',fontsize=FONTSIZE+2)
        sky.field_image(fig,ax2,dec_field)
        plt.show()

    ## Light Recover
    light_results = {}
    rec_lum, Drec_lum = sky.light_recover(dec_field,thr,bkg_mean,(ker_sigma,Dker_sigma),extraction=light_results,binning=BINNING,results=results,display_fig=results)
    params['objects'] = light_results
    params['dec_field'] = dec_field

    mean_lum = stars.mean_lum() if stars.mean_lum() is not None else stars.lum.mean()           #: mean value
    # average and compute the STD
    mean_rec = np.mean(rec_lum)
    observable = stars.lum>bkg_mean
    # print
    print(f'Slum {stars.lum[:4]}')
    print(f'Rlum {rec_lum[:4]}')
    print(f'S: {mean_lum:.2e}\tL: {mean_rec:.2e}\t{(mean_rec-mean_lum)/mean_lum:.2%}')
    print(f'FOUND: {len(rec_lum)}')
    print(f'EXPEC: {len(stars.lum[observable])}')

    if results:
        plt.figure()
        plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
        plt.hist(rec_lum,BINNING,histtype='step',color='blue')
        plt.axvline(mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed',alpha=0.5)
        # plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
        plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed',alpha=0.5)
        # plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dashed')
        plt.legend(fontsize=FONTSIZE)
        plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
        plt.ylabel('counts',fontsize=FONTSIZE)
        plt.grid(linestyle='dashed',color='gray',alpha=0.5)
        plt.show()
    if checks:
        plt.figure()
        plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
        bins = np.linspace(min(stars.lum.min(),rec_lum.min()),max(stars.lum.max(),rec_lum.max()),BINNING*3)
        plt.hist(stars.lum,bins,density=True,color='red',histtype='step')
        plt.hist(rec_lum,bins,density=True,histtype='step',color='blue')
        plt.axvline(mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed',alpha=0.5)
        plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed',alpha=0.5)
        # plt.axvline(new_thr,0,1,color='green',label='threshold',linestyle='dashed',alpha=0.5)
        # plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
        plt.legend(fontsize=FONTSIZE)
        plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
        plt.ylabel('counts',fontsize=FONTSIZE)
        plt.grid(linestyle='dashed',color='gray',alpha=0.5)
        plt.show()
    return (rec_lum, Drec_lum), params

def check_results(rec_lum: sky.NDArray, Drec_lum: sky.NDArray, parameters: dict, observable: list[bool], stars: sky.Star,frame_size: int = sky.FRAME['size'], star_num: int = sky.FRAME['stars'], mass_range: tuple[float,float] = sky.FRAME['mass_range'], mass_seed: float | None = None, pos_seed: float | None = None, bkg_seed: float | None = None, det_seed: float | None = None, bkg_param: tuple = sky.field.BACK_PARAM, overlap: bool = True, acq_num: int = 6, montecarlo: bool = True, iter: int = ITERATIONS, checks:bool=True) -> tuple[tuple[sky.NDArray,sky.NDArray,sky.NDArray],tuple[sky.NDArray,sky.NDArray,sky.NDArray], list[bool],tuple[sky.NDArray,sky.NDArray,sky.NDArray]]:
    bkg_mean      = parameters['bkg']['mean']
    ker_sigma     = parameters['kernel']['sigma']
    dec_field     = parameters['dec_field']
    light_results = parameters['objects']

    rec_lum = np.copy(rec_lum)
    Drec_lum = np.copy(Drec_lum)
    ## Position check
    init_pos = np.asarray(stars.pos)[:,observable]      #: source stars positions
    rec_pos  = np.copy(light_results['pos'])                  #: recovered stars positions
    # store the variable for distance check
    rec_pos0 = np.copy(light_results['pos'])                  
    # compute distances between true and recovered
    check = np.array([ sky.distance(rec_pos,init_pos[:,i]) for i in range(init_pos.shape[1])])
    s0_cnts = len(check[check ==   0])                  #: counts of matched objects
    s1_cnts = len(check[check <=   ker_sigma])          #: counts within 1 kernel sigma
    s2_cnts = len(check[check <= 2*ker_sigma])          #: counts within 2 kernel sigma
    s3_cnts = len(check[check <= 3*ker_sigma])          #: counts within 3 kernel sigma

    print('COUNTS')
    print('s0',s0_cnts,'-->',s0_cnts/len(rec_lum)*100,'%')
    print('s1',s1_cnts,'-->',s1_cnts-s0_cnts,(s1_cnts-s0_cnts)/len(rec_lum)*100,'%')
    print('s2',s2_cnts,'-->',s2_cnts-s1_cnts,(s2_cnts-s1_cnts)/len(rec_lum)*100,'%')
    print('s3',s3_cnts,'-->',s3_cnts-s2_cnts,(s3_cnts-s2_cnts)/len(rec_lum)*100,'%')

    rec_num = len(rec_lum)      #: number of recovered objects
    if checks:
        WIDTH = 0.2
        plt.figure()
        plt.title('Positions comparison',fontsize=FONTSIZE+2)
        plt.bar(np.arange(4)/4,[s0_cnts,s1_cnts,s2_cnts,s3_cnts],width=WIDTH)
        plt.xticks(np.arange(4)/4,['$0\\sigma$','$1\\sigma$','$2\\sigma$','$3\\sigma$'])
        plt.ylabel('counts',fontsize=FONTSIZE)
        plt.grid(linestyle='dashed',color='gray',alpha=0.3)

        plt.figure()
        plt.title('Positions comparison',fontsize=FONTSIZE+2)
        plt.bar(np.arange(4)/4,[s0_cnts,s1_cnts-s0_cnts,s2_cnts-s1_cnts,s3_cnts-s2_cnts],width=WIDTH)
        plt.xticks(np.arange(4)/4,['$0\\sigma$','$1\\sigma$','$2\\sigma$','$3\\sigma$'])
        plt.ylabel('eff. counts',fontsize=FONTSIZE)
        plt.grid(linestyle='dashed',color='gray',alpha=0.3)
        # plt.show()

        plt.figure()
        plt.title('Positions comparison',fontsize=FONTSIZE+2)
        plt.bar(np.arange(4)/4,[s0_cnts/rec_num*100,(s1_cnts-s0_cnts)/rec_num*100,(s2_cnts-s1_cnts)/rec_num*100,(s3_cnts-s2_cnts)/rec_num*100],width=WIDTH)
        plt.xticks(np.arange(4)/4,['$0\\sigma$','$1\\sigma$','$2\\sigma$','$3\\sigma$'])
        plt.ylabel('eff. counts [%]',fontsize=FONTSIZE)
        plt.grid(linestyle='dashed',color='gray',alpha=0.3)
        plt.show()

    ## Brightness check 
    # define variables to store values
    fake_index = []                                 #: artifacts
    new_sort = np.empty(0)                          #: source objects
    tmp_init_pos = np.arange(init_pos.shape[1])     #: position counter
    for i in range(rec_num):
        try:
            print(f'{i} - {tmp_init_pos}',end='\r')
            # compute distances
            obj_dist = sky.distance(rec_pos[:,i],init_pos[:,tmp_init_pos])
            # find the minimum
            mindist  = np.argmin(obj_dist)
            if obj_dist[mindist] > 2*ker_sigma:
                # artifact
                fake_index += [i]
            elif obj_dist[mindist] != 0:
                # look for other source objects
                upper = (obj_dist[mindist] // ker_sigma + 1) * ker_sigma 
                sect  = obj_dist <= upper 
                if len(obj_dist[sect]) != 1:
                    init_lum = stars.lum[tmp_init_pos].copy()
                    # take the brightest 
                    minpos = np.argmax(init_lum[sect])
                # store
                new_sort = np.append(new_sort,tmp_init_pos[mindist])
                # tmp_init_pos = np.delete(tmp_init_pos,mindist)  
            else:
                # store
                new_sort = np.append(new_sort,tmp_init_pos[mindist])
                # remove the object
                tmp_init_pos = np.delete(tmp_init_pos,mindist)  
        except:
            print()
            print(mindist,new_sort,tmp_init_pos)
            raise

    new_sort = new_sort.astype(int)
    print()
    print(mindist,new_sort,tmp_init_pos)
    print('LEN',len(new_sort),len(np.unique(new_sort)))
    # update the arrays
    fake_lum  = rec_lum[fake_index]
    Dfake_lum = Drec_lum[fake_index]
    fake_pos  = rec_pos[:,fake_index]
    rec_lum  = np.delete(rec_lum,fake_index)
    Drec_lum = np.delete(Drec_lum,fake_index)
    rec_pos  = np.delete(rec_pos,fake_index,axis=1)
    # source brightness
    lum0 = stars.lum[new_sort]
    # compute the differences
    diff = rec_lum-lum0
    for lum,Dlum,l0,sl in zip(rec_lum,Drec_lum,lum0,diff):
        print(f'{l0*1e3:.1f} - [{lum*1e3:.1f} +- {Dlum*1e3:.1f}] -> {abs(sl)/l0:.2%} l0 -> {abs(sl)/Dlum:.2} sigma')
    # remove artifacts
    fake_index = abs(diff/Drec_lum) > 3
    # update the arrays
    fake_lum  = np.append(fake_lum,rec_lum[fake_index])
    Dfake_lum = np.append(Dfake_lum,Drec_lum[fake_index])
    fake_pos  = np.append(fake_pos,rec_pos[:,fake_index],axis=1)
    lum0 = np.delete(lum0,fake_index)
    diff = np.delete(diff,fake_index)
    rec_lum  = np.delete(rec_lum,fake_index)
    Drec_lum = np.delete(Drec_lum,fake_index)
    rec_pos  = np.delete(rec_pos,fake_index,axis=1)
    # plot
    if checks:
        plt.figure()
        x_ticks = np.arange(len(rec_lum))*10
        plt.title('Brightness comparison',fontsize=FONTSIZE+2)
        plt.errorbar(x_ticks,diff,Drec_lum,fmt='.--',capsize=3)
        plt.axhline(0,0,1,color='k')
        plt.grid(linestyle='dashed',color='gray',alpha=0.5)
        plt.xticks(x_ticks,np.round(rec_lum*1e3,1))
        plt.xlabel('$\\ell_{rec}$ [$10^{-3}$]',fontsize=FONTSIZE)
        plt.ylabel('$\\ell_{rec} - \\ell_0$',fontsize=FONTSIZE)

        plt.figure()
        plt.hist(diff/Drec_lum,10)

        fig,ax = plt.subplots(1,1)
        sky.field_image(fig,ax,dec_field)
        ax.plot(init_pos[1,new_sort],init_pos[0,new_sort],'xg',label='detected')
        ax.plot(*np.delete(init_pos,new_sort,axis=1)[::-1],'xr')
        ax.plot(*rec_pos[::-1],'.b')
        ax.plot(*fake_pos[::-1],'.r',label='artifacts')
        ax.legend()
        plt.show()

    distances, rec_distances, rec_distances0 = None, None , None
    ## Distances Check
    if montecarlo:
        distances = np.array([ generate_sample(selection=bkg_mean,acq_num=acq_num,dim=frame_size,stnum=star_num,masses=mass_range,back_param=bkg_param,back_seed=bkg_seed,det_seed=det_seed,overlap=overlap,seed=(mass_seed,pos_seed)) for _ in range(iter) ]).flatten()
        rec_distances = sky.dist_corr(rec_pos)
        rec_distances0 = sky.dist_corr(rec_pos0)
        if checks:
            plt.figure()
            plt.suptitle('Distribution of objects relative distances',fontsize=FONTSIZE+2)
            plt.subplot(121)
            plt.title('No artifacts removal',fontsize=FONTSIZE+2)
            plt.xlabel('$d$ [px]',fontsize=FONTSIZE)
            plt.ylabel('norm. counts',fontsize=FONTSIZE)
            g_cnts, bins, _ = plt.hist(distances,BINNING*3,color='red',density=True,histtype='step',label='generated')
            rec_cnts, _, _  = plt.hist(rec_distances0,bins,color='blue',density=True,histtype='step',label='recover')
            plt.legend(fontsize=FONTSIZE)
            plt.subplot(122)
            plt.title('Artifacts removal',fontsize=FONTSIZE+2)
            plt.xlabel('$d$ [px]',fontsize=FONTSIZE)
            # plt.ylabel('norm. counts',fontsize=FONTSIZE)
            g_cnts, bins, _ = plt.hist(distances,BINNING*3,color='red',density=True,histtype='step',label='generated')
            rec_cnts, _, _  = plt.hist(rec_distances,bins,color='blue',density=True,histtype='step',label='recover')
            plt.legend(fontsize=FONTSIZE)
            plt.show()
    return (rec_lum, Drec_lum, rec_pos), (fake_lum, Dfake_lum, fake_pos), lum0, (distances, rec_distances, rec_distances0)

def pipeline(frame_size: int = sky.FRAME['size'], star_num: int = sky.FRAME['stars'], mass_range: tuple[float,float] = sky.FRAME['mass_range'], mass_seed: float | None = None, pos_seed: float | None = None, bkg_seed: float | None = None, det_seed: float | None = None, bkg_param: tuple = sky.field.BACK_PARAM, overlap: bool = True, acq_num: int = 6, montecarlo: bool = True, iter: int = ITERATIONS, display_plots: bool = False, results:bool=True, verbose_display: bool = False, checks:bool=True,stop_ctrl:bool = False, checks_trigger: bool = True, save: dict | None = None, log: dict | None = None) -> tuple[list[sky.NDArray],list[sky.NDArray],tuple[sky.Star, sky.NDArray], list[sky.NDArray | None]] | tuple[list[sky.NDArray], dict, sky.Star]:
    from time import time
    start_time = time()
    bkg_val = bkg_param[1][0]*1e4*sky.K
    
    if save is not None:
        if 'id' not in save.keys():
            save['id'] = 0

    if log is not None:
        log_name  = log['file_name'] if 'file_name' in log.keys() else ''
        directory = log['main_dir']  if 'main_dir'  in log.keys() else ''
        id_num = save['id'] if save is not None else 0
        log_update(f'MEAN VAL:\t{bkg_val} - {id_num:02d}',file_name=log_name,main_dir=directory)
    ctrl_cnt = 0
    while True:
        ### SCIENCE FRAME
        STARS, (sci_frame, Dsci_frame) = science_frame(frame_size, star_num, mass_range, mass_seed, pos_seed, bkg_seed, det_seed, bkg_param, overlap, acq_num, display_plots, results)

        ### RESTORATION
        (rec_lum, Drec_lum), params = restoration(sci_frame,Dsci_frame,STARS,display_plots,results,verbose_display,checks,stop_ctrl)
        
        if ctrl_cnt == 20: 
            if log is not None:
               log_update(f"\t!FAIL:\tMore than 20 iterations",file_name=log_name,main_dir=directory)
            
            break

        try:
            if save is not None:
                save_dir = save['main_dir']
                object_pos = params['objects']['pos']
                name = f"bkg-{bkg_val:.2f}_{save['id']:02d}"
                sky.store_results(name,[rec_lum,Drec_lum,object_pos[0],object_pos[1]],main_dir=save_dir,columns=['L','DL','X','Y'])
            break
        except ValueError as valerr:
            ctrl_cnt += 1
            if log is not None:
               log_update(f"\t!FAIL:\tlen rec {len(rec_lum)}\tDrec {len(Drec_lum)}",file_name=log_name,main_dir=directory)
               log_update('\tError: '+str(valerr),file_name=log_name,main_dir=directory)


    if log is not None:
        end_time = time()
        bkg = params['bkg']
        ker = params['kernel']
        log_update(f"\tBackgr:\t{bkg['mean']}\t{bkg['sigma']}",file_name=log_name,main_dir=directory)
        log_update(f"\tKernel:\t{ker['sigma']}\t{ker['Dsigma']}",file_name=log_name,main_dir=directory)
        log_update(f"Time:\t{end_time-start_time}",file_name=log_name,main_dir=directory)
        log_update('=='*30+'\n',file_name=log_name,main_dir=directory)


    ### CHECKS
    if checks_trigger:
        observable = STARS.lum > params['bkg']['mean']
        (rec_lum, Drec_lum, rec_pos), (fake_lum, Dfake_lum, fake_pos), lum0, (distances, rec_distances, rec_distances0) = check_results(rec_lum,Drec_lum,params,observable,STARS,frame_size,star_num,mass_range,mass_seed,pos_seed,bkg_seed,det_seed,bkg_param,overlap,acq_num,montecarlo,iter,checks)
        print('\n- - END - -')
        print('\n'+'=='*30+'\n\n')
        return [rec_lum,Drec_lum,rec_pos], [fake_lum,Dfake_lum,fake_pos], (STARS, lum0), [distances, rec_distances, rec_distances0]
    else:
        print('\n- - END - -')
        print('\n'+'=='*30+'\n\n')
        return [rec_lum, Drec_lum], params, STARS


if __name__ == '__main__':
    
    ## Constants
    DISPLAY_PLOTS = False

    params = DEFAULT_PARAMS.copy()



    # mseeds = [ 16, 18, 19, 27, 29, 46, 48 ]

    # params.pop('mass_seed')
    # results = [pipeline(mass_seed=ms,**params,overlap=True,results=False,montecarlo=True) for ms in mseeds]

    ### DEFAULT WITH OVERLAP 
    # _ = pipeline(**DEFAULT_PARAMS,overlap=True)

    ### MULTIPLE RUNS
    # params['pos_seed'] = None
    # mult_data = [ pipeline(**params,overlap=None,results=False,checks=False)[-1] for _ in range(10)]
    # distances = np.concatenate([ data[0] for data in mult_data])
    # rec_distances  = np.concatenate([ data[1] for data in mult_data])
    # rec_distances0 = np.concatenate([ data[2] for data in mult_data])
    # plt.figure()
    # plt.suptitle('Distribution of objects relative distances',fontsize=FONTSIZE+2)
    # plt.subplot(121)
    # plt.title('No artifacts removal',fontsize=FONTSIZE+2)
    # plt.xlabel('$d$ [px]',fontsize=FONTSIZE)
    # plt.ylabel('norm. counts',fontsize=FONTSIZE)
    # g_cnts, bins, _ = plt.hist(distances,BINNING*3,color='red',density=True,histtype='step',label='generated')
    # rec_cnts, _, _  = plt.hist(rec_distances0,bins,color='blue',density=True,histtype='step',label='recover')
    # plt.legend(fontsize=FONTSIZE)
    # plt.subplot(122)
    # plt.title('Artifacts removal',fontsize=FONTSIZE+2)
    # plt.xlabel('$d$ [px]',fontsize=FONTSIZE)
    # # plt.xlabel('norm. counts',fontsize=FONTSIZE)
    # g_cnts, bins, _ = plt.hist(distances,BINNING*3,color='red',density=True,histtype='step',label='generated')
    # rec_cnts, _, _  = plt.hist(rec_distances,bins,color='blue',density=True,histtype='step',label='recover')
    # plt.legend(fontsize=FONTSIZE)
    # plt.show()

    # ### SPARSELY POPULATED 
    # params['mass_seed'] += 3
    # params['pos_seed']  += 3
    # _ = pipeline(star_num=50,**params,overlap=True,iter=3000)
    
    # ### OVER-POPULATED 
    # params['mass_seed'] += 3
    # params['pos_seed']  += 3
    # _ = pipeline(star_num=500,**params,overlap=True, iter=1000)

    ### DIFFERNT BACKGROUNDS
    # bkg_mean = np.linspace(3.5,5.0,10)
    # params['bkg_seed'] = None
    # BKG_ITER = 5

    # bkg_data = [pipeline(**params,overlap=True,bkg_param=('Gaussian',(bkg*1e-4/sky.K,bkg*1e-4/sky.K*20e-2)),results=False,checks=False,montecarlo=False)[:-1] for bkg in bkg_mean]

    # # store data
    # DIRECTORY = 'multi-means'
    # NAME = 'source'
    # NAMES = [f'bkg-{bkg:.2f}' for bkg in bkg_mean]
    # HEADER = ['r-a','Dra','w-a','Dwa']
    # STARS = bkg_data[0][2][0]
    # sky.store_results('source',[STARS.m,STARS.lum,STARS.pos[0],STARS.pos[1]],main_dir=DIRECTORY,columns=['M','L','X','Y'])
    # for name, bkg in zip(NAMES,bkg_data):
    #     print('Save '+name)
    #     diff = len(bkg[0][0]) - len(bkg[1][0])
    #     ext1 = 0 if diff >= 0 else -diff
    #     ext2 = 0 if diff <= 0 else diff
    #     col1 = np.append(bkg[0][0],[-1]*ext1)
    #     col2 = np.append(bkg[0][1],[-1]*ext1)
    #     col3 = np.append(bkg[1][0],[-1]*ext2)
    #     col4 = np.append(bkg[1][1],[-1]*ext2)
    #     print('\tdiff -',diff)
    #     print('\tcol1 -',len(col1))
    #     print('\tcol2 -',len(col2))
    #     print('\tcol3 -',len(col3))
    #     print('\tcol4 -',len(col4))
    #     if len(col1) != len(col3):
    #         print(col1)
    #         print(col2)
    #         raise
    #     elif len(col1) != len(col2):
    #         print(col1)
    #         print(col2)
    #         raise
    #     elif len(col3) != len(col4):
    #         print(col1)
    #         print(col2)
    #         raise
    #     sky.store_results(name,[col1,col2,col3,col4],main_dir=DIRECTORY,columns=HEADER)

    # print(len(bkg_data),len(bkg_data[0]),len(bkg_data[0][0]))
    # for i in range(len(bkg_data)):
    #     bkg = bkg_data[i]
    #     print(f'{i} - elment')
    #     print('\t',bkg[0][0])
    #     print('\t',bkg[1][0])
    #     print('\t',bkg[2][0])
    #     if len(bkg[0][0]) != 0:
    #         print('\t\t',np.mean(bkg[0][0])) 
    #     else: 
    #         print('\t\tNada')
    # print([np.mean(bkg[0][0]) if len(bkg[0][0]) != 0 else 0 for bkg in bkg_data])
    # rec_lum = np.array([np.mean(bkg[0][0]) if len(bkg[0][0]) != 0 else 0 for bkg in bkg_data])
    # rec_lum_tot = np.array([np.append(bkg[0][0],bkg[1][0]).mean() if len(bkg[0][0]) != 0 else np.mean(bkg[1][0]) for bkg in bkg_data])
    # sour_lum = np.array([bkg[2][0].mean_lum() for bkg in bkg_data])
    # plt.figure()
    # plt.plot(bkg_mean,rec_lum-sour_lum,'.--b',label='rem. art.')
    # plt.plot(bkg_mean,rec_lum_tot-sour_lum,'.--r',label='with art.')
    # plt.axhline(0,0,1,color='k')
    # plt.legend()
    # plt.xlabel('$\\bar{n}_B$')
    # plt.ylabel('$\\bar{\\ell}_{rec} - \\bar{\\ell}_{0}$')
    # plt.grid()
    # plt.show()


    ### MALQUIST BIAS
    BKG_VALUES = [11,15,20]
    BKG_ITER = 20
    params['bkg_seed'] = None
    DIRECTORY = 'multi-bkg-real'
    LOG_NAME = 'multi_bkg'

    # log_update('Multi Bkg different realizations\n',file_name=LOG_NAME,main_dir=DIRECTORY,mode='w')

    # [ [[[rec_lum, Drec_lum], params, STARS], ...], ...]
    _ = np.asarray([ [ pipeline(**params,overlap=True,bkg_param=('Gaussian',(bkg*1e-4/sky.K,bkg*1e-4/sky.K*20e-2)),results=False,checks_trigger=False,checks=False,log={'file_name': LOG_NAME,'main_dir': DIRECTORY},save={'main_dir': DIRECTORY,'id': i}) for i in range(BKG_ITER)] for bkg in BKG_VALUES],dtype='object')
    # # recovered  = [ [ [rec_lum, Drec_lum], ... ], ... ]
    # # parameters = [ [  params, ... ], ... ]
    # # stars      = [ [  STARS, ... ], ... ]
    # recovered  = SAMPLES[:,:,0]
    # parameters = SAMPLES[:,:,1]
    # stars      = SAMPLES[:,:,2]
    # del SAMPLES
    # stars = stars[0][0]
    # avg_lum = stars.mean_lum()
    # sky.store_results('source',[stars.m,stars.lum,stars.pos[0],stars.pos[1]],main_dir=DIRECTORY,columns=['M','L','X','Y'])

    # # rec = [[rec_lum, Drec_lum], ...]
    # # par = [params, ...]
    # for j in range(len(BKG_VALUES)):
    #     val = BKG_VALUES[j]
    #     log_update(f'MEAN VAL:\t{val}',file_name=LOG_NAME,main_dir=DIRECTORY)
    #     for i in range(BKG_ITER):
    #         pr = parameters[j,i]
    #         rc = recovered[j,i]
    #         bkg = pr['bkg']
    #         ker = pr['kernel']
    #         log_update(f"Backgr:\t{bkg['mean']}\t{bkg['sigma']}",file_name=LOG_NAME,main_dir=DIRECTORY)
    #         log_update(f"Kernel:\t{ker['sigma']}\t{ker['Dsigma']}",file_name=LOG_NAME,main_dir=DIRECTORY)

    #         avg_rc = np.mean(rc[0])
    #         object_pos = pr['objects']['pos']
    #         name = f'bkg-{val:.2f}-{avg_rc*1e2:.1f}_{i:02d}'
    #         sky.store_results(name,[rc[0],rc[1],object_pos[0],object_pos[1]],main_dir=DIRECTORY,columns=['L','DL','X','Y'])

    #         log_update('=='*30+'\n',file_name=LOG_NAME,main_dir=DIRECTORY)
        
