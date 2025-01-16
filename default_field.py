import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky

## Constants
DISPLAY_PLOTS = False
BINNING = 63
FONTSIZE = 18

### MONTE CARLO REALIZATIONS
def generate_sample() -> sky.NDArray:
    _, S = sky.initialize(p_seed=None)
    distances = sky.dist_corr(S.pos)
    return distances

max_iter = 3000
distances = np.array([ generate_sample() for _ in range(max_iter) ])
dist_means = np.array([np.mean(d) for d in distances])
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

## Science Frame
sci_frame = master_light - master_dark 
Dsci_frame = np.sqrt(Dmaster_light**2 + Dmaster_dark**2)
sky.fast_image(sci_frame,'Science Frame')

### RESTORATION
## Background Estimation
(bkg_mean, _), bkg_sigma = sky.bkg_est(sci_frame,display_plot=True)

## Kernel Estimation
thr = bkg_mean + bkg_sigma  #: the threshold for searching algorithm
max_num_obj = 10            #: number of objects at the most 
# extract objects
objs, Dobjs, objs_pos = sky.searching(sci_frame,thr,bkg_mean,Dsci_frame,num_objs=max_num_obj,cntrl=20,log=True)

# fit a 2DGaussian profile to extraction
ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0),results=True,title='title-only')
# compute the kernel
atm_kernel = sky.Gaussian(sigma=ker_sigma)

## RL Algorithm
dec_field = sky.LR_deconvolution(sci_frame,atm_kernel,Dsci_frame,bkg_mean,bkg_sigma)

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.set_title('Before deconvolution',fontsize=20)
sky.field_image(fig,ax1,sci_frame)
ax2.set_title('After deconvolution',fontsize=20)
sky.field_image(fig,ax2,dec_field)
plt.show()

## Light Recover
results = {}
rec_lum, Drec_lum = sky.light_recover(dec_field,thr,bkg_mean,(ker_sigma,Dker_sigma),results=results,BINNING=BINNING)

lum = np.sort(STARS.lum)[::-1]  #: initial brightness values 
mean_lum = lum.mean()           #: mean value
# average and compute the STD
mean_rec, Dmean_rec = sky.mean_n_std(rec_lum)
# print
sky.print_measure(mean_rec,Dmean_rec,'L')
print(f'S: {mean_lum:.2e}\t{(mean_rec-mean_lum)/Dmean_rec:.2f}')
print(f'S: {mean_lum:.2e}\tL: {mean_rec:.2e}\t{(mean_rec-mean_lum)/mean_lum:.2%}')

## Plots
# try:
#     ratio = int(rec_lum.max()/rec_lum.min())
#     bins  = int(len(rec_lum) / np.log10(ratio))*2 if ratio != 1 else int(len(rec_lum)*3/4)
# except OverflowError:
#     print(rec_lum.max(),rec_lum.min())
#     bins = int(len(rec_lum)*3/4)
plt.figure()
plt.hist(rec_lum,BINNING,histtype='step')
plt.axvline(mean_rec,0,1,label='mean recovered brightness')
plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
plt.axvline(mean_lum,0,1,color='red',label='mean source brightness')
plt.legend(fontsize=FONTSIZE)
plt.show()

rec_distances = sky.dist_corr(results['pos'])
plt.figure()
g_cnts, bins, _ = plt.hist(distances,BINNING,density=True,histtype='step',label='generated')
rec_cnts, _, _  = plt.hist(rec_distances,bins,density=True,histtype='step',label='recover')
plt.legend(fontsize=FONTSIZE)

plt.figure()
cen_bins = (bins[1:] + bins[:-1])/2
plt.plot(cen_bins,rec_cnts-g_cnts,'.--')
plt.show()
