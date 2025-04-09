import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky

## Constants
PLOTTING = True
FONTSIZE = 18
DIR_NAME = 'poisson_det'

### INITIALIZATION
## Parameters
BINNING = 20

source_frame, STARS = sky.initialize(overlap=True,display_fig=True,norm='log')
print(len(source_frame[source_frame<0]))
conv_frame = sky.atm_seeing(source_frame,sigma=sky.SEEING_SIGMA,bkg=0,display_fig=True)
print(len(conv_frame[conv_frame<0]))
xpos, ypos = np.where(conv_frame<0)
plt.figure()
plt.imshow(conv_frame+conv_frame.min()+1,norm='log')
plt.plot(ypos,xpos,'.r')
plt.show()
conv_frame[conv_frame<0] = 0
light_frame = conv_frame

plt.figure()
plt.hist(light_frame.flatten(),100)
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2)
sky.field_image(fig,ax1,conv_frame,norm='log')
sky.field_image(fig,ax2,light_frame,norm='log')
plt.show()
## Kernel Estimation
thr = 0                     #: the threshold for searching algorithm
max_num_obj = 10            #: number of objects at the most 
obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
# extract objects
objs, Dobjs, _ = sky.searching(light_frame,thr,0,np.zeros(light_frame.shape),ker_sigma=None,num_objs=max_num_obj,cntrl=20,log=True,obj_params=obj_param,display_fig=True)
# fit a 2DGaussian profile to extraction
ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(0,0),obj_param=obj_param,results=True,title='title-only')
# compute the kernel
atm_kernel = sky.Gaussian(sigma=ker_sigma)

from skimage.restoration import richardson_lucy
dec_field = richardson_lucy(light_frame,atm_kernel.kernel(),2000)
## RL Algorithm
# dec_field = sky.LR_deconvolution(light_frame,atm_kernel,np.zeros(light_frame.shape),0,0,max_r=2000,results=True,display_fig=True,norm='log')

print('DEC',len(dec_field[dec_field<0]))
print('DEC',len(dec_field[dec_field==0]))

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.set_title('Before deconvolution',fontsize=FONTSIZE+2)
sky.field_image(fig,ax1,light_frame,norm='log')
ax2.set_title('After deconvolution',fontsize=FONTSIZE+2)
sky.field_image(fig,ax2,dec_field)
# sky.field_image(fig,ax2,dec_field,norm='log')
plt.show()
## Light Recover
results = {}
obj_params = [[],[],[],[]]
rec_lum, Drec_lum = sky.light_recover(dec_field,thr,0,(ker_sigma,Dker_sigma),extraction=results,binning=BINNING,results=PLOTTING,display_fig=PLOTTING,object_params=obj_params,relax_cond=True)

mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()           #: mean value
# average and compute the STD
mean_rec = np.mean(rec_lum)
observable = slice(None,None) #STARS.lum>bkg_mean
# print
print(f'Slum {STARS.lum[:4]}')
print(f'Rlum {rec_lum[:4]}')
print(f"Rmmm {[obj.max() for obj in results['objs'][:4]]}")
print(f'Rmmm {obj_params[0][:4]}')
print(f'Rmmm {np.asarray(obj_params[0][:4])*np.sqrt(2*np.pi)}')
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

plt.figure()
plt.plot(rec_lum[:min(len(rec_lum),100)]-STARS.lum[:min(len(rec_lum),100)],'.--')
plt.figure()
plt.plot((rec_lum[:min(len(rec_lum),100)]-STARS.lum[:min(len(rec_lum),100)])/STARS.lum[:min(len(rec_lum),100)]*100,'.--')
plt.figure()
plt.plot(rec_lum[:min(len(rec_lum),100)]/STARS.lum[:min(len(rec_lum),100)],'.--')

plt.show()
