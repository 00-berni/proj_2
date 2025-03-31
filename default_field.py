import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky

## Constants
PLOTTING = True
DISPLAY_PLOTS = False
BINNING = 20
FONTSIZE = 18
BKG_MEAN = sky.BACK_MEAN
OVERLAP  = True

### MONTE CARLO REALIZATIONS
def generate_sample(selection: None | sky.ArrayLike = None,**initargs) -> sky.NDArray:
    _, S = sky.initialize(**initargs,p_seed=None)
    pos = np.asarray(S.pos)
    if selection is not None:
        pos = pos[:,S.lum > selection]
    distances = sky.dist_corr(pos)
    return distances

max_iter = 2000
from time import time    
start_time = time()
distances = np.array([ generate_sample(overlap=OVERLAP) for _ in range(max_iter) ])
end_time = time()
print('TIME:',end_time-start_time)
samples = distances.copy()
dist_means = np.mean(distances,axis=1)
distances = distances.flatten()

mean_dist = np.mean(dist_means)
std_dist = np.std(dist_means)

print('MONTE CARLO REALIZATIONS')
print('Mean distance:',np.mean(dist_means),np.std(dist_means))
if PLOTTING:
    plt.figure()
    plt.suptitle(f'{max_iter} realizations',fontsize=FONTSIZE+3)
    plt.subplot(1,2,1)
    plt.title('Distance distribution',fontsize=FONTSIZE)
    plt.hist(distances,BINNING*3,density=True,histtype='step')
    plt.axvline(np.mean(distances),0,1,color='orange',linestyle='dashed',label='mean')
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel('distance [px]',fontsize=FONTSIZE)
    plt.ylabel('norm. counts',fontsize=FONTSIZE)
    plt.subplot(1,2,2)
    plt.title('Distribution of distance mean',fontsize=FONTSIZE)
    plt.hist(dist_means,BINNING*3,density=True,histtype='step')
    plt.axvline(mean_dist,0,1,color='orange',linestyle='dashed',label='mean')
    plt.axvspan(mean_dist-std_dist,mean_dist+std_dist,facecolor='orange',label='STD',alpha=0.2)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel('mean distance [px]',fontsize=FONTSIZE)
    plt.ylabel('norm. counts',fontsize=FONTSIZE)
    plt.show()

### SCIENCE FRAME
## Initialization
STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(overlap=OVERLAP,results=PLOTTING,display_fig=DISPLAY_PLOTS)
if PLOTTING:
    STARS.plot_info()
    plt.show()
print('CHECK',sky.BACK_MEAN*sky.K,len(STARS.lum[STARS.lum<sky.BACK_MEAN*sky.K]),STARS.lum.max(),STARS.lum[:4])
## Science Frame
sci_frame = master_light - master_dark 
Dsci_frame = np.sqrt(Dmaster_light**2 + Dmaster_dark**2)
if PLOTTING:
    sky.fast_image(sci_frame,'Science Frame')
    plt.figure()
    plt.hist(Dsci_frame.flatten(),71)
    plt.show()

### RESTORATION
## Background Estimation
(bkg_mean, _), bkg_sigma = sky.bkg_est(sci_frame,display_plot=PLOTTING)

## Kernel Estimation
thr = bkg_mean + bkg_sigma  #: the threshold for searching algorithm
max_num_obj = 10            #: number of objects at the most 
obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
# extract objects
objs, Dobjs, objs_pos = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,ker_sigma=None,num_objs=max_num_obj,cntrl=20,log=True,obj_params=obj_param,display_fig=PLOTTING)
print(obj_param)
# fit a 2DGaussian profile to extraction
ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0),obj_param=obj_param,results=PLOTTING,title='title-only')
# compute the kernel
atm_kernel = sky.Gaussian(sigma=ker_sigma)

## RL Algorithm
dec_field = sky.LR_deconvolution(sci_frame,atm_kernel,Dsci_frame,bkg_mean,bkg_sigma,results=PLOTTING)

if PLOTTING:
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.set_title('Before deconvolution',fontsize=FONTSIZE+2)
    sky.field_image(fig,ax1,sci_frame)
    ax2.set_title('After deconvolution',fontsize=FONTSIZE+2)
    sky.field_image(fig,ax2,dec_field)
    plt.show()

## Light Recover
results = {}
rec_lum, Drec_lum = sky.light_recover(dec_field,thr,bkg_mean,(ker_sigma,Dker_sigma),extraction=results,binning=BINNING,results=PLOTTING,display_fig=PLOTTING)

mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()           #: mean value
# average and compute the STD
mean_rec = np.mean(rec_lum)
observable = STARS.lum>bkg_mean
# print
print(f'Slum {STARS.lum[:4]}')
print(f'Rlum {rec_lum[:4]}')
print(f'S: {mean_lum:.2e}\tL: {mean_rec:.2e}\t{(mean_rec-mean_lum)/mean_lum:.2%}')
print(f'FOUND: {len(rec_lum)}')
print(f'EXPEC: {len(STARS.lum[observable])}')

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
plt.figure()
plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
bins = np.linspace(min(STARS.lum.min(),rec_lum.min()),max(STARS.lum.max(),rec_lum.max()),BINNING*3)
plt.hist(STARS.lum,bins,density=True,color='red',histtype='step')
plt.hist(rec_lum,bins,density=True,histtype='step',color='blue')
plt.axvline(mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed',alpha=0.5)
plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed',alpha=0.5)
# plt.axvline(new_thr,0,1,color='green',label='threshold',linestyle='dashed',alpha=0.5)
# plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
plt.legend(fontsize=FONTSIZE)
plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
plt.ylabel('norm. counts',fontsize=FONTSIZE)
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.show()

### CHECKS
## Position check
init_pos = np.asarray(STARS.pos)[:,observable]      #: source stars positions
rec_pos  = np.copy(results['pos'])                  #: recovered stars positions
# store the variable for distance check
rec_pos0 = np.copy(results['pos'])                  
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

rec_num = len(rec_lum)      #: number of recovered objects
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
                init_lum = STARS.lum[tmp_init_pos].copy()
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
print('Rec0',rec_lum.shape,Drec_lum.shape,rec_pos.shape)
fake_obj = rec_lum[fake_index]
fake_pos = rec_pos[:,fake_index]
rec_lum  = np.delete(rec_lum,fake_index)
Drec_lum = np.delete(Drec_lum,fake_index)
rec_pos  = np.delete(rec_pos,fake_index,axis=1)
print('Rec1',rec_lum.shape,Drec_lum.shape,rec_pos.shape)
# source brightness
lum0 = STARS.lum[new_sort]
# compute the differences
diff = rec_lum-lum0
for lum,Dlum,l0,sl in zip(rec_lum,Drec_lum,lum0,diff):
    print(f'{l0*1e3:.1f} - [{lum*1e3:.1f} +- {Dlum*1e3:.1f}] -> {abs(sl)/l0:.2%} l0 -> {abs(sl)/Dlum:.2} sigma')
# remove artifacts
fake_index = abs(diff/Drec_lum) > 3
# update the arrays
fake_obj = np.append(fake_obj,rec_lum[fake_index])
fake_pos = np.append(fake_pos,rec_pos[:,fake_index],axis=1)
lum0 = np.delete(lum0,fake_index)
diff = np.delete(diff,fake_index)
rec_lum  = np.delete(rec_lum,fake_index)
Drec_lum = np.delete(Drec_lum,fake_index)
rec_pos  = np.delete(rec_pos,fake_index,axis=1)
# plot
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

## Distances Check
obs_distances = np.array([ generate_sample(selection=bkg_mean) for _ in range(max_iter) ]).flatten()
rec_distances = sky.dist_corr(rec_pos)
rec_distances0 = sky.dist_corr(rec_pos0)
plt.figure()
plt.suptitle('Distribution of objects relative distances',fontsize=FONTSIZE+2)
plt.subplot(121)
plt.title('No artifacts removal',fontsize=FONTSIZE+2)
plt.xlabel('$d$ [px]',fontsize=FONTSIZE)
plt.ylabel('norm. counts',fontsize=FONTSIZE)
g_cnts, bins, _ = plt.hist(distances,BINNING*3,color='red',density=True,histtype='step',label='generated')
rec_cnts, _, _  = plt.hist(rec_distances0,bins,color='blue',density=True,histtype='step',label='recover')
obs_cnts, _, _  = plt.hist(obs_distances,bins,color='green',density=True,histtype='step',label='observable')
plt.legend(fontsize=FONTSIZE)
plt.subplot(122)
plt.title('Artifacts removal',fontsize=FONTSIZE+2)
plt.xlabel('$d$ [px]',fontsize=FONTSIZE)
# plt.ylabel('norm. counts',fontsize=FONTSIZE)
g_cnts, bins, _ = plt.hist(distances,BINNING*3,color='red',density=True,histtype='step',label='generated')
rec_cnts, _, _  = plt.hist(rec_distances,8,color='blue',density=True,histtype='step',label='recover')
obs_cnts, _, _  = plt.hist(obs_distances,bins,color='green',density=True,histtype='step',label='observable')
plt.legend(fontsize=FONTSIZE)

# def compare(gen_sample: sky.NDArray, data_cnts: sky.NDArray, binning: sky.NDArray, avg_dist: float, und_pop: list[float], over_pop: list[float]):
#     smpl_cnts, _, = np.histogram(gen_sample,binning,density=True)
#     avg_bin = (binning[1:]+binning[:-1])/2
#     diff = data_cnts - smpl_cnts
#     near_pos = np.argmax(abs(diff[avg_bin < avg_dist]))
#     far_pos  = np.argmax(abs(diff[avg_bin > avg_dist])) + np.where(avg_bin > avg_dist)[0][0]
#     near_max = avg_bin[near_pos]
#     far_max  = avg_bin[far_pos]
#     if diff[near_pos] < 0:   und_pop  += [near_max]
#     elif diff[near_pos] > 0: over_pop += [near_max]
#     if diff[far_pos] < 0:    und_pop  += [far_max]
#     elif diff[far_pos] > 0:  over_pop += [far_max]
#     return near_max, far_max

# und_pop  = []
# over_pop = []
# near, far = np.array([ compare(sample,rec_cnts,bins,mean_dist,und_pop,over_pop) for sample in samples]).T

# und_pop  = np.asarray(und_pop).flatten()
# over_pop = np.asarray(over_pop).flatten()

# plt.figure()
# plt.title('Near')
# plt.hist(near,31,density=True)
# plt.figure()
# plt.title('Far')
# plt.hist(far,31,density=True)
# plt.figure()
# plt.title('Under pop')
# plt.hist(und_pop,31,density=True)
# plt.figure()
# plt.title('Over pop')
# plt.hist(over_pop,31,density=True)
# plt.show()

## Distribution Check
mean_lum = np.mean(rec_lum)
plt.figure()
plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
bins = np.linspace(min(STARS.lum.min(),rec_lum.min()),max(STARS.lum.max(),rec_lum.max()),BINNING*3)
plt.hist(STARS.lum,bins,density=True,color='red',histtype='step')
plt.hist(rec_lum,bins,density=True,histtype='step',color='blue')
plt.axvline(mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed',alpha=0.5)
plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed',alpha=0.5)
# plt.axvline(new_thr,0,1,color='green',label='threshold',linestyle='dashed',alpha=0.5)
# plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
plt.legend(fontsize=FONTSIZE)
plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
plt.ylabel('norm. counts',fontsize=FONTSIZE)
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.show()


exit()
### ANOTHER METHOD
g_objs, g_errs, g_pos = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,ker_sigma=None,max_size=10,cntrl=None,cntrl_sel='bright',sel_cond='new',debug_plots=False,log=True)
g_rec_lum  = np.array([ obj.max() for obj in g_objs])
g_Drec_lum = np.array([ err[sky.peak_pos(obj)] for obj,err in zip(g_objs,g_errs)])

g_Drec_lum = g_rec_lum * np.sqrt(2*np.pi)*ker_sigma * np.sqrt((g_Drec_lum/g_rec_lum)**2 + (Dker_sigma/ker_sigma)**2)
g_rec_lum *= np.sqrt(2*np.pi)*ker_sigma
g_rec_lum = np.array([ obj.sum() for obj in g_objs])

mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()           #: mean value
# average and compute the STD
g_mean_rec = np.mean(g_rec_lum)
# print
print(f'Slum {STARS.lum[:4]}')
print(f'Glum {g_rec_lum[:4]}')
print(f'S: {mean_lum:.2e}\tL: {g_mean_rec:.2e}\t{(g_mean_rec-mean_lum)/mean_lum:.2%}')
print(f'FOUND: {len(g_rec_lum)}')
print(f'EXPEC: {len(STARS.lum[STARS.lum>bkg_mean])}')

plt.figure()
plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
plt.hist(g_rec_lum,BINNING,histtype='step',color='blue')
plt.axvline(g_mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed')
# plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed')
# plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dashed')
plt.legend(fontsize=FONTSIZE)
plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
plt.ylabel('counts',fontsize=FONTSIZE)
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.figure()
plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
bins = np.linspace(min(STARS.lum.min(),g_rec_lum.min()),max(STARS.lum.max(),g_rec_lum.max()),BINNING)
plt.hist(STARS.lum,bins,density=True,color='red',histtype='step')
plt.hist(g_rec_lum,bins,density=True,histtype='step',color='blue')
plt.axvline(g_mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed')
plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dashed')
# plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed')
plt.legend(fontsize=FONTSIZE)
plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
plt.ylabel('counts',fontsize=FONTSIZE)
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.show()

g_diff = g_rec_lum-STARS.lum[:len(g_rec_lum)]
plt.figure()
plt.title('Brightness comparison',fontsize=FONTSIZE+2)
plt.errorbar(np.arange(len(g_rec_lum)),g_diff,g_Drec_lum,fmt='.--',capsize=3)
plt.axhline(0,0,1,color='k')
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.xticks(np.arange(len(g_rec_lum)),np.round(g_rec_lum*1e4,0))
plt.xlabel('$\\ell_{rec}$',fontsize=FONTSIZE)
plt.ylabel('$\\ell_{rec} - \\ell_0$',fontsize=FONTSIZE)

plt.figure()
plt.hist(g_diff/g_Drec_lum,30)
plt.show()


init_pos = np.asarray(STARS.pos)
g_rec_pos  = np.copy(g_pos)
g_check = np.array([ sky.distance(g_rec_pos,init_pos[:,i]) for i in range(init_pos.shape[1])])
s1_cnts = len(g_check[g_check <=   ker_sigma])/init_pos.shape[1]*100
s2_cnts = len(g_check[g_check <= 2*ker_sigma])/init_pos.shape[1]*100
s3_cnts = len(g_check[g_check <= 3*ker_sigma])/init_pos.shape[1]*100

WIDTH = 0.25
plt.figure()
plt.bar(np.arange(3)/3,[s1_cnts,s2_cnts,s3_cnts],width=WIDTH)
plt.xticks(np.arange(3)/3,['$1\\sigma$','$2\\sigma$','$3\\sigma$'])
plt.show()

rec_distances = sky.dist_corr(g_pos)
plt.figure()
plt.title('Distribution of objects relative distances',fontsize=FONTSIZE+2)
plt.xlabel('$d$ [px]',fontsize=FONTSIZE)
plt.xlabel('norm. counts',fontsize=FONTSIZE)
g_cnts, bins, _ = plt.hist(distances,BINNING,color='red',density=True,histtype='step',label='generated')
rec_cnts, _, _  = plt.hist(rec_distances,bins,color='blue',density=True,histtype='step',label='recover')
plt.legend(fontsize=FONTSIZE)

def compare(gen_sample: sky.NDArray, data_cnts: sky.NDArray, binning: sky.NDArray, avg_dist: float, und_pop: list[float], over_pop: list[float]):
    smpl_cnts, _, = np.histogram(gen_sample,binning,density=True)
    avg_bin = (binning[1:]+binning[:-1])/2
    diff = data_cnts - smpl_cnts
    near_pos = np.argmax(abs(diff[avg_bin < avg_dist]))
    far_pos  = np.argmax(abs(diff[avg_bin > avg_dist])) + np.where(avg_bin > avg_dist)[0][0]
    near_max = avg_bin[near_pos]
    far_max  = avg_bin[far_pos]
    if diff[near_pos] < 0:   und_pop  += [near_max]
    elif diff[near_pos] > 0: over_pop += [near_max]
    if diff[far_pos] < 0:    und_pop  += [far_max]
    elif diff[far_pos] > 0:  over_pop += [far_max]
    return near_max, far_max


und_pop  = []
over_pop = []
near, far = np.array([ compare(sample,rec_cnts,bins,mean_dist,und_pop,over_pop) for sample in samples]).T

und_pop  = np.asarray(und_pop).flatten()
over_pop = np.asarray(over_pop).flatten()

plt.figure()
plt.title('Near')
plt.hist(near,31,density=True)
plt.figure()
plt.title('Far')
plt.hist(far,31,density=True)
plt.figure()
plt.title('Under pop')
plt.hist(und_pop,31,density=True)
plt.figure()
plt.title('Over pop')
plt.hist(over_pop,31,density=True)
plt.show()
