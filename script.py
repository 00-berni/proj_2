import numpy as np
from numpy import correlate
import matplotlib.pyplot as plt
import skysimulation.display as display
import skysimulation.field as field
import skysimulation.restoration as restore
from skysimulation.field import NDArray, K, MIN_m, BETA, SEEING_SIGMA

def autocorr(vec: restore.Sequence, mode: str = 'same') -> restore.NDArray:
    return correlate(vec,vec,mode)

if __name__ == '__main__':
    S, (m_light, s_light), (m_dark, s_dark) = field.field_builder()
    # N = 100
    # M = 100
    # figure = False
    # norm = 'linear'
    # back = field.BACK_PARAM
    # det = field.NOISE_PARAM

    # print('\n--- Build the Field ---')
    # S, I = field.field_builder(N,M,back_param=back,det_param=det,display_fig=figure,norm=norm)
    # S.plot_info(field.ALPHA,field.BETA)

    # print('\n--- Dark ---')
    # dark = restore.dark_elaboration(det,dim=N)

    # print('\n--- Background Estimation ---')
    # bkg = restore.bkg_est(I,True)
    # print(bkg/field.K)

    # print('\n--- Objects Extraction ---')
    # thr = 10
    # mean_val = max(bkg,dark.mean())
    # err, mean_val = restore.err_estimation(I,mean_val,thr=thr,display_plot=True)

    # print(f'MEAN ERR = {mean_val} +- {err}\t{err/mean_val}')
    # objs, obj_pos = restore.object_isolation(I,mean_val,size=7,objnum=15,acc=err/mean_val,reshape=True,reshape_corr=False,sel_cond=True,minsize=2,grad_new=False,display_fig=False,norm=norm, corr_cond=True)

    # coor = np.array([*S.pos])
    # ind  = np.where(S.lum > mean_val)[0]
    # fig, ax = plt.subplots(1,1)
    # display.field_image(fig,ax,I)
    # if len(obj_pos[1]) != 0:
    #     ax.plot(obj_pos[1],obj_pos[0],'.b')
    # ax.plot(coor[1,ind],coor[0,ind],'x', color='yellow')
    # ax.plot(coor[1,:ind.min()],coor[0,:ind.min()],'x', color='violet')

    # plt.show()



    # print('\n--- Kernel Estimation ---')
    # if objs is not None:
    #     m = mean_val
    #     s = err / np.sqrt(2*np.log(2))
    #     for obj in objs:
    #         art_noise = np.random.normal(m,s,obj.shape)
    #         fig,ax = plt.subplots(1,1)
    #         display.field_image(fig,ax,obj)
    #         fig,ax = plt.subplots(1,1)
    #         display.field_image(fig,ax,art_noise)
    #         rows = np.vstack(obj)
    #         nrows = np.vstack(art_noise)
    #         plt.figure()
    #         for i in range(len(rows)):
    #             plt.plot(np.correlate(rows[i],nrows[i],'same'),'.-',label=f'{i}')
    #         plt.legend()
    #         plt.show()
    
    #     print('MEAN VAL',mean_val)
    #     kernel, (sigma, Dsigma) = restore.kernel_estimation(objs,err,N,all_results=True,display_plot=False)
    #     rec_I = restore.LR_deconvolution(I,kernel,mean_val,iter=20,sel='rl',display_fig=True)
    #     mask = restore.mask_filter(rec_I,I,True)
    #     lum, pos, allpos = restore.find_objects(rec_I,I,kernel,mean_val,sel_pos=obj_pos,acc=1e-1,res_str=['lum','acc','pos'],display_fig=False)
    #     l, Dl = lum

    #     # index  = pos.copy()
    #     # index0 = np.array(S.pos)

    #     # min_dim = min(len(index[0]),len(index0[0]))
    #     # dx, dy = abs(index[:,:min_dim] - index0[:,:min_dim])
    #     # discr = 4
    #     # mat_pos = np.where(np.logical_and(dx<=discr,dy<=discr))[0]        
    #     # matches = len(mat_pos)

    #     biggest = np.where(S.lum > mean_val)[0]
    #     big_num = len(biggest)

    #     print('\n----------------\n')
    #     print(f'MAX MASS - {field.MAX_m}')
    #     print(f'Found:\t{len(l)} / {M}')
    #     print(f'Precision:\t{len(l)/M*100:.2f} %')
    #     print(f'Biggest:\t{big_num}')
    #     print(f'Precision:\t{len(l)/big_num*100:.2f} %')

    #     star_pos = np.array(S.pos)
    #     fig, ax = plt.subplots(1,1)
    #     display.field_image(fig,ax,rec_I)
    #     ax.plot(star_pos[1],star_pos[0],'.',color='blue')
    #     ax.plot(star_pos[1,biggest],star_pos[0,biggest],'s',color='green')
    #     ax.plot(pos[1],pos[0],'x',color='red')
    #     plt.show()

    #     # print(f'Match:\t{matches} / {M}')
    #     # print(f'Accuracy:\t{matches/len(l)*100:.2f} %')

    #     # obj = objs[3]
    #     # for iter in [30,80,100,200]:
    #     #     rec_obj = restore.LR_deconvolution(obj,kernel,mean_val,iter=iter,sel='rl',display_fig=True)
    #     #     x,y = restore.peak_pos(rec_obj)
    #     #     m = len(obj)//2 - 2
    #     #     xx = slice(x-m,x+m+1)
    #     #     yy = slice(y-m,y+m+1)
    #     #     rec_obj[xx,yy] = 0
    #     #     field.fast_image(rec_obj,title=f'Iter {iter}')
    #     #     _ = restore.mask_filter(restore.LR_deconvolution(I,kernel,mean_val,iter=iter,sel='rl',display_fig=True),I,True)

    # else:
    #     print('[ALERT] - It is not possible to recover the field!\nTry to change parameters')
