import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky

## Constants
DISPLAY_PLOTS = False
BINNING = 63
FONTSIZE = 18

### MONTE CARLO REALIZATIONS
def generate_sample(**initargs) -> sky.NDArray:
    _, S = sky.initialize(**initargs,p_seed=None)
    distances = sky.dist_corr(S.pos)
    return distances

max_iter = 2000
from time import time    
start_time = time()
distances = np.array([ generate_sample() for _ in range(max_iter) ])
end_time = time()
print('TIME:',end_time-start_time)
samples = distances.copy()
dist_means = np.mean(distances,axis=1)
distances = distances.flatten()

mean_dist = np.mean(dist_means)
std_dist = np.std(dist_means)

print('MONTE CARLO REALIZATIONS')
print('Mean distance:',np.mean(dist_means),np.std(dist_means))
plt.figure()
plt.suptitle(f'{max_iter} realizations',fontsize=FONTSIZE+3)
plt.subplot(1,2,1)
plt.title('Distance distribution',fontsize=FONTSIZE)
plt.hist(distances,BINNING,density=True,histtype='step')
plt.axvline(np.mean(distances),0,1,color='orange',linestyle='dashed',label='mean')
plt.legend(fontsize=FONTSIZE)
plt.xlabel('distance [px]',fontsize=FONTSIZE)
plt.ylabel('norm. counts',fontsize=FONTSIZE)
plt.subplot(1,2,2)
plt.title('Distribution of distance mean',fontsize=FONTSIZE)
plt.hist(dist_means,BINNING,density=True,histtype='step')
plt.axvline(mean_dist,0,1,color='orange',linestyle='dashed',label='mean')
plt.axvspan(mean_dist-std_dist,mean_dist+std_dist,facecolor='orange',label='STD',alpha=0.2)
plt.legend(fontsize=FONTSIZE)
plt.xlabel('mean distance [px]',fontsize=FONTSIZE)
plt.ylabel('norm. counts',fontsize=FONTSIZE)
plt.show()

### SCIENCE FRAME
## Initialization
STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(display_fig=DISPLAY_PLOTS)
# STARS.plot_info(sky.ALPHA,sky.BETA)
# plt.show()
print('CHECK',sky.BACK_MEAN*sky.K,len(STARS.lum[STARS.lum<sky.BACK_MEAN*sky.K]),STARS.lum.max(),STARS.lum[:4])
## Science Frame
sci_frame = master_light - master_dark 
Dsci_frame = np.sqrt(Dmaster_light**2 + Dmaster_dark**2)
sky.fast_image(sci_frame,'Science Frame')
plt.figure()
plt.hist(Dsci_frame.flatten(),71)
plt.show()

### RESTORATION
## Background Estimation
(bkg_mean, _), bkg_sigma = sky.bkg_est(sci_frame,display_plot=True)

## Kernel Estimation
thr = bkg_mean + bkg_sigma  #: the threshold for searching algorithm
max_num_obj = 10            #: number of objects at the most 
obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
# extract objects
objs, Dobjs, objs_pos = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,num_objs=max_num_obj,cntrl=20,log=True,obj_params=obj_param)
print(obj_param)
# fit a 2DGaussian profile to extraction
ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0),obj_param=obj_param,results=True,title='title-only')
# compute the kernel
atm_kernel = sky.Gaussian(sigma=ker_sigma)

### ANOTHER METHOD
obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
g_objs, g_errs, g_pos = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,ker_sigma=ker_sigma,max_size=10,cntrl=None,cntrl_sel='bright',sel_cond='new',obj_params=obj_param,debug_plots=False,log=True)
g_rec_lum  = np.array([ obj.max() for obj in g_objs])
g_Drec_lum = np.array([ err[sky.peak_pos(obj)] for obj,err in zip(g_objs,g_errs)])
obj_param = np.asarray(obj_param)
g_Drec_lum = g_rec_lum * np.sqrt(2*np.pi)*ker_sigma * np.sqrt((g_Drec_lum/g_rec_lum)**2 + (Dker_sigma/ker_sigma)**2)
g_rec_lum *= np.sqrt(2*np.pi)*ker_sigma
g_rec_lum = np.array([ obj.sum() for obj in g_objs])
sortpos = np.argsort(g_rec_lum)[::-1]
g_rec_lum = g_rec_lum[sortpos]
g_objs = [g_objs[i] for i in sortpos]
g_errs = [g_errs[i] for i in sortpos]
g_pos  = g_pos[:,sortpos]

mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()           #: mean value
# average and compute the STD
g_mean_rec = np.mean(g_rec_lum)
# print
print(f'Slum {STARS.lum[:4]}')
print(f'Glum {g_rec_lum[:4]}')
print(f'G2lum {obj_param[0,:4,0]}')
print(f'S: {mean_lum:.2e}\tL: {g_mean_rec:.2e}\t{(g_mean_rec-mean_lum)/mean_lum:.2%}')
print(f'FOUND: {len(g_rec_lum)}')
print(f'EXPEC: {len(STARS.lum[STARS.lum>bkg_mean])}')

plt.figure()
plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
plt.hist(g_rec_lum,BINNING,histtype='step',color='blue')
plt.axvline(g_mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dotted')
# plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dotted')
# plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dotted')
plt.legend(fontsize=FONTSIZE)
plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
plt.ylabel('counts',fontsize=FONTSIZE)
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.figure()
plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
bins = np.linspace(min(STARS.lum.min(),g_rec_lum.min()),max(STARS.lum.max(),g_rec_lum.max()),BINNING)
plt.hist(STARS.lum,bins,density=True,color='red',histtype='step')
plt.hist(g_rec_lum,bins,density=True,histtype='step',color='blue')
plt.axvline(g_mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dotted')
plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dotted')
# plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dotted')
plt.legend(fontsize=FONTSIZE)
plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
plt.ylabel('counts',fontsize=FONTSIZE)
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.show()

g_diff = g_rec_lum-STARS.lum[:len(g_rec_lum)]
plt.figure()
x_ticks = np.arange(len(g_rec_lum))*10
plt.title('Brightness comparison',fontsize=FONTSIZE+2)
plt.errorbar(x_ticks,g_diff,g_Drec_lum,fmt='.--',capsize=3)
plt.axhline(0,0,1,color='k')
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.xticks(x_ticks,np.round(g_rec_lum*1e3,1))
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
