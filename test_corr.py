# import numpy as np
# from numpy import correlate
# import matplotlib.pyplot as plt
# from skysimulation import NDArray
# from skysimulation import field
# from skysimulation import display
# from skysimulation import restoration as restore
import test_rl as test
from test_rl import np, NDArray, plt, autocorr, correlate, corr_plot
from test_rl import restore as rst
from test_rl import display as dspl

def extr_obj(field: NDArray, pos: NDArray | tuple[NDArray,NDArray], null_pos: NDArray | tuple[NDArray,NDArray], thr: float, size: int = 7, acc: float = 0.1, display_fig: bool = False, **kwargs) -> tuple[list[NDArray] | None, list[NDArray] | None]:
    tmp_field = np.copy(field)
    a_pos = np.empty(shape=(2,0),dtype=int)
    keys = np.array(list(kwargs.keys()))
    sel_keys = {'mindist','minsize','cutsize','reshape'}
    sel_kwarg = {k: kwargs[k] for key in sel_keys for k in keys[np.where(keys==key)[0]]} if len(keys) != 0 else {}
    fig_keys = kwargs.keys() - sel_keys
    fig_kwarg = {k: kwargs[k] for key in fig_keys for k in keys[np.where(keys==key)[0]]} if len(keys) != 0 else {}
    extraction = []
    for x,y in zip(*pos):
        peak = tmp_field[x,y]
        if peak <= thr:
            break
        a_size = rst.grad_check(tmp_field,(x,y),thr,size,acc)

        xu, xd, yu, yd = a_size.flatten()
        xr = slice(x-xd, x+xu+1)
        yr = slice(y-yd, y+yu+1)
        tmp_field[xr,yr] = 0.0
        a_pos = np.append(a_pos,[[x],[y]],axis=1)
        if not any(([[0,0],[0,0]] == a_size).all(axis=1)):
            obj = field[xr,yr].copy()
            save_cond = rst.selection(obj,(x,y),a_pos,size,sel='all',acc=acc,**sel_kwarg)
            if save_cond:
                extraction += [obj]
                if display_fig:
                    title = f'Accepted - ({x},{y})'
            else:
                if display_fig:
                    title = f'Rejected - ({x},{y})'
            if display_fig:
                fig, ax = plt.subplots(1,1)
                ax.set_title(title)
                dspl.field_image(fig,ax,tmp_field,**fig_kwarg)
                dspl.fast_image(obj,title,**fig_kwarg)
        else:
            if display_fig:
                fig, ax = plt.subplots(1,1)
                ax.set_title('Super Rejected')
                dspl.field_image(fig,ax,tmp_field,**fig_kwarg)
                plt.show()
    noise = []
    for x,y in zip(*null_pos):
        peak = tmp_field[x,y]
        if peak <= thr:
            break
        a_size = rst.grad_check(tmp_field,(x,y),thr,size,acc)

        xu, xd, yu, yd = a_size.flatten()
        xr = slice(x-xd, x+xu+1)
        yr = slice(y-yd, y+yu+1)
        tmp_field[xr,yr] = 0.0
        a_pos = np.append(a_pos,[[x],[y]],axis=1)
        if not any(([[0,0],[0,0]] == a_size).all(axis=1)):
            obj = field[xr,yr].copy()
            save_cond = rst.selection(obj,(x,y),a_pos,size,sel='all',acc=acc,**sel_kwarg)
            if save_cond:
                noise += [obj]
                if display_fig:
                    title = f'Noise - Accepted - ({x},{y})'
            else:
                if display_fig:
                    title = f'Noise - Rejected - ({x},{y})'
            if display_fig:
                fig, ax = plt.subplots(1,1)
                ax.set_title(title)
                dspl.field_image(fig,ax,tmp_field,**fig_kwarg)
                dspl.fast_image(obj,title,**fig_kwarg)
    if len(extraction) == 0: extraction = None
    if len(noise) == 0: noise = None
    return extraction, noise





if __name__ == '__main__':
    N = 100
    M = 5
    set_pos = np.array([[N//2,10,76,2,45],
                        [N//2,9,66,33,80]])
    max_mass = 5
    F, S = test.initialize(N,M,max_mass,'set',set_pos,display_fig=True)
    I = test.add_effects(F,S.m,S.pos)
    dspl.fast_image(I)
    nul_pos = np.empty((2,0),int)
    xd = [20,41,60,60,75,75,00,26,51,76]
    xu = [40,59,75,75,99,99,25,50,75,99]
    yd = [00,00,00,20,00,20,88,88,88,88]
    yu = [20,20,20,40,20,40,99,99,99,99]
    for id,iu,jd,ju in zip(xd,xu,yd,yu):
        xr = slice(id,iu)    
        yr = slice(jd,ju)    
        cut = I[xr,yr].copy()
        x,y = rst.peak_pos(cut)
        x += id
        y += jd
        nul_pos = np.append(nul_pos,[[x],[y]],axis=1)
    sortind = np.argsort(I[nul_pos[0],nul_pos[1]])
    nul_pos = nul_pos[:,sortind]
    fig, ax = plt.subplots(1,1)
    dspl.field_image(fig,ax,I)
    ax.plot(set_pos[1],set_pos[0],'.b')
    ax.plot(nul_pos[1],nul_pos[0],'.r')
    plt.show()

    bkg = rst.bkg_est(I,True)
    dark = rst.dark_elaboration(test.field.NOISE_PARAM)
    mean_val = max(bkg,dark.mean())
    err, mean_val = rst.err_estimation(I,mean_val,10,display_plot=False)
    objs, noises = extr_obj(I,set_pos[:,::-1],nul_pos,mean_val,display_fig=False)

    # Noise
    m = mean_val
    s = err / np.sqrt(2*np.log(2))
    mm = np.empty(0,float)
    print('\nNOISE - ART NOISE Corr')
    from scipy.signal import find_peaks
    for n in noises:
        hcorr = autocorr(n.flatten())
        vcorr = autocorr(np.stack(n,axis=-1).flatten())
        hpeak, _ = find_peaks(hcorr)
        vpeak, _ = find_peaks(vcorr)
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(n.flatten())
        plt.subplot(2,2,2)
        plt.plot(hcorr,'.-')    
        if len(hpeak) > 0:
            plt.plot(hpeak,hcorr[hpeak],'x',color='red')
        plt.subplot(2,2,3)
        plt.plot(np.stack(n,axis=-1).flatten())
        plt.subplot(2,2,4)
        plt.plot(vcorr,'.-')
        if len(vpeak) > 0:
            plt.plot(vpeak,vcorr[vpeak],'x',color='red')
        plt.figure()
        plt.plot(autocorr(np.append(n,np.stack(n,axis=-1))),'.-')    
        plt.show()
        print('\n---Lenghts---')
        print(f'{len(hpeak)}\t{len(vpeak)}')
    for obj in objs:
        hcorr = autocorr(obj.flatten())
        vcorr = autocorr(np.stack(obj,axis=-1).flatten())
        hpeak, _ = find_peaks(hcorr)
        vpeak, _ = find_peaks(vcorr)
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(obj.flatten())
        plt.subplot(2,2,2)
        plt.plot(hcorr,'.-')    
        if len(hpeak) > 0:
            plt.plot(hpeak,hcorr[hpeak],'x',color='red')
        plt.subplot(2,2,3)
        plt.plot(np.stack(obj,axis=-1).flatten())
        plt.subplot(2,2,4)
        plt.plot(vcorr,'.-')    
        if len(vpeak) > 0:
            plt.plot(vpeak,vcorr[vpeak],'x',color='red')
        plt.figure()
        plt.plot(autocorr(np.append(obj,np.stack(obj,axis=-1))),'.-')    
        plt.show()
        print('\n---Lenghts---')
        print(f'{len(hpeak)}\t{len(vpeak)}')
    # for n in noises:
    #     art_noise = np.random.normal(m,s,n.shape)
    #     m_corr, a_ncorr, a_corr, diff = corr_plot(n,art_noise)
    #     mm = np.append(mm,np.mean(diff))
    # plt.figure()
    # plt.plot(mm,'.--')
    # plt.show()
    # mm = np.empty(0,float)
    # print('\nSIGNAL - ART NOISE Corr')
    # for obj in objs:
    #     art_noise = np.random.normal(m,s,obj.shape)
    #     m_corr, a_ncorr, a_corr, diff = corr_plot(obj,art_noise)
    #     mm = np.append(mm,np.mean(diff))
    # plt.figure()
    # plt.plot(mm,'.--')
    # plt.show()

