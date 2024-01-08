import numpy as np
import matplotlib.pyplot as plt
import skysimulation.display as display
import skysimulation.field as field
import skysimulation.restoration as restore


if __name__ == '__main__':
    N = 100
    M = 4500
    figure = False
    norm = 'linear'
    back = field.BACK_PARAM
    det = field.NOISE_PARAM

    print('\n--- Build the Field ---')
    S, I = field.field_builder(N,M,back_param=back,det_param=det,display_fig=figure,norm=norm)
    S.plot_info(field.ALPHA,field.BETA)

    print('\n--- Dark ---')
    dark = restore.dark_elaboration(det,dim=N)

    print('\n--- Background Estimation ---')
    bkg = restore.bkg_est(I,True)
    print(bkg/field.K)

    print('\n--- Objects Extraction ---')
    mean_val = max(bkg,dark.mean())
    objs, obj_pos = restore.object_isolation(I,mean_val,size=7,objnum=15,reshape=True,reshape_corr=True,sel_cond=True,display_fig=False,norm=norm)

    print('\n--- Kernel Estimation ---')
    if objs is not None:
        thr = 10
        err = restore.err_estimation(I,mean_val,thr=thr,display_plot=True)
        kernel, (sigma, Dsigma) = restore.kernel_estimation(objs,err,N,all_results=True,display_plot=True)
        rec_I = restore.LR_deconvolution(I,kernel,mean_val,iter=50,sel='rl',display_fig=True)

        dim = len(rec_I)
        size = np.zeros((4,2))
        cond = np.array([True]*4)
        for i in range(dim-1):
            diff1 = rec_I[i+1,i+1] - rec_I[i,i]
            diff2 = rec_I[dim-1-i-1,dim-1-i-1] - rec_I[dim-1-i,dim-1-i]
            diff3 = rec_I[i+1,dim-1-i-1] - rec_I[i,dim-1-i]
            diff4 = rec_I[dim-1-i-1,i+1] - rec_I[dim-1-i,i]
            if diff1 < 0:
                size[0] = [i]*2
                cond[0] = False
            if diff2 < 0:
                size[1] = [dim-1-i]*2
                cond[1] = False
            if diff3 < 0:
                size[2] = [i,dim-1-i]
                cond[2] = False
            if diff4 < 0:
                size[3] = [dim-1-i,i]
                cond[3] = False
            if all(cond): break
        size = size.max(axis=0)
        print('size',size)

    else:
        print('[ALERT] - It is not possible to recover the field!\nTry to change parameters')
