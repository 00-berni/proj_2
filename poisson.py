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
DC = sky.MIN_m**sky.BETA*sky.K / 7
M_SEED = 48
POI_SEED = 100
ACQ_NUM = 6

source_frame, STARS = sky.initialize(m_seed=M_SEED,overlap=True,display_fig=True,norm='log')
print(len(source_frame[source_frame<0]))
clear_conv_frame = sky.atm_seeing(source_frame,sigma=sky.SEEING_SIGMA,bkg=0,display_fig=True) 
conv_frame = clear_conv_frame // DC
print(len(conv_frame[conv_frame<0]))
xpos, ypos = np.where(conv_frame<0)
plt.figure()
plt.imshow(conv_frame+conv_frame.min()+1,norm='log')
plt.plot(ypos,xpos,'.r')
plt.show()
conv_frame[conv_frame<0] = 0
rng = np.random.default_rng(seed=POI_SEED)
light_frame = np.array([[ rng.poisson(conv_frame[i,j],ACQ_NUM) for j in range(sky.N)]for i in range(sky.N)]).T*DC

if ACQ_NUM > 1:
    cols = ACQ_NUM//2 if ACQ_NUM%2==0 else ACQ_NUM//2+1
    fig, axs = plt.subplots(2,cols)
    colorbar = {'colorbar': False, 'colorbar_pos': 'bottom'}
    for i in range(ACQ_NUM):
        index = (i//cols,i%cols)
        # if i%2 == 1: colorbar['colorbar'] = True  
        axs[index].set_title(f'Light Frame {i}',fontsize=20)
        if i == ACQ_NUM-1: colorbar['colorbar'] = True
        sky.field_image(fig, axs[index],light_frame[i],**colorbar)
        # colorbar['colorbar'] = False
    plt.show()

conv_frame *= DC 
master_light = np.mean(light_frame,axis=0).T
print('LIGHT',len(master_light[master_light<0]))


sky.fast_image(master_light,'Master Light')
sky.fast_image(master_light,'Master Light',norm='log')
plt.figure()
plt.hist(master_light.flatten(),100)
plt.show()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.set_title('Source + Seeing',fontsize=FONTSIZE)
sky.field_image(fig,ax1,clear_conv_frame)
ax2.set_title('Master Light',fontsize=FONTSIZE)
sky.field_image(fig,ax2,master_light)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.set_title('Source + Seeing',fontsize=FONTSIZE)
sky.field_image(fig,ax1,clear_conv_frame,norm='log')
ax2.set_title('Master Light',fontsize=FONTSIZE)
sky.field_image(fig,ax2,master_light,norm='log')
plt.show()
## Kernel Estimation
thr = 0                     #: the threshold for searching algorithm
max_num_obj = 10            #: number of objects at the most 
obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
# extract objects
objs, Dobjs, _ = sky.searching(master_light,thr,0,np.zeros(master_light.shape),ker_sigma=None,num_objs=max_num_obj,cntrl=20,log=True,obj_params=obj_param,display_fig=True)
# fit a 2DGaussian profile to extraction
ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(0,0),obj_param=obj_param,results=True,title='title-only')
# compute the kernel
atm_kernel = sky.Gaussian(sigma=ker_sigma)

from skimage.restoration import richardson_lucy
dec_field = richardson_lucy(master_light,atm_kernel.kernel(),1000)
## RL Algorithm
# dec_field = sky.LR_deconvolution(light_frame,atm_kernel,np.zeros(light_frame.shape),0,0,max_r=2000,results=True,display_fig=True,norm='log')

print('DEC',len(dec_field[dec_field<0]))
print('DEC',len(dec_field[dec_field==0]))

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.set_title('Before deconvolution',fontsize=FONTSIZE+2)
sky.field_image(fig,ax1,master_light)
ax2.set_title('After deconvolution',fontsize=FONTSIZE+2)
sky.field_image(fig,ax2,dec_field)
plt.show()

sky.fast_image(dec_field,'Deconvolved Field',norm='log',colorbar=False)

## Light Recover
results = {}
rec_lum, Drec_lum = sky.light_recover(dec_field,thr,0,(ker_sigma,Dker_sigma),extraction=results,binning=BINNING,results=PLOTTING,display_fig=PLOTTING,relax_cond=True,debug_plots=False,gausfit=False)

mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()           #: mean value
# average and compute the STD
mean_rec = np.mean(rec_lum)
observable = slice(None,None) #STARS.lum>bkg_mean
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
plt.xscale('log')
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
plt.xscale('log')
plt.grid(linestyle='dashed',color='gray',alpha=0.5)
plt.show()



# LAMBDA = 5
# # LAMBDA = sky.BACK_MEAN*sky.K * 50
# K = 1
# DET_PARAM = ('Poisson',(LAMBDA,K))

# # built the field
# STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(quantize=DC,det_param=DET_PARAM,results=PLOTTING,display_fig=False)

# # science_frame  = master_light - master_dark
# # Dscience_frame = np.sqrt(Dmaster_light**2 + Dmaster_dark**2)
# science_frame  = master_light
# Dscience_frame = Dmaster_light

# print(STARS.mean_lum(),STARS.lum.max(),STARS.lum.min())

# print('NEGATIVE',len(np.where(master_dark <0)[0]) )
# print('NEGATIVE',len(np.where(science_frame <0)[0]) )

# print(np.var(master_light))
# plt.figure(1)
# plt.hist(Dmaster_light.flatten(),1000,histtype='step')
# plt.figure(2)
# cnts, bins,_ = plt.hist(master_light.flatten(),500,histtype='step')
# plt.xscale('log')
# plt.yscale('log')
# plt.figure(3)
# cnts, bins,_ = plt.hist(science_frame.flatten(),500,histtype='step')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# ### RESTORATION
# ## Background Estimation
# bkg_mean, bkg_sigma = sky.bkg_est(science_frame,display_plot=PLOTTING)

# ## Kernel Estimation
# thr = bkg_mean + bkg_sigma  #: the threshold for searching algorithm
# max_num_obj = 10            #: number of objects at the most 
# obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
# # extract objects
# objs, Dobjs, _ = sky.searching(science_frame,thr,bkg_mean,Dscience_frame,ker_sigma=None,num_objs=max_num_obj,cntrl=20,log=True,obj_params=obj_param,display_fig=PLOTTING)
# # fit a 2DGaussian profile to extraction
# ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0),obj_param=obj_param,results=PLOTTING,title='title-only')
# # compute the kernel
# atm_kernel = sky.Gaussian(sigma=ker_sigma)

# ## RL Algorithm
# dec_field = sky.LR_deconvolution(science_frame,atm_kernel,np.zeros(science_frame.shape),bkg_mean,bkg_sigma,max_r=2000,results=PLOTTING,display_fig=True,norm='log')

# if PLOTTING:
#     fig,(ax1,ax2) = plt.subplots(1,2)
#     ax1.set_title('Before deconvolution',fontsize=FONTSIZE+2)
#     sky.field_image(fig,ax1,science_frame)
#     ax2.set_title('After deconvolution',fontsize=FONTSIZE+2)
#     sky.field_image(fig,ax2,dec_field)
#     plt.show()


# ## Light Recover
# light_results = {}
# rec_lum, Drec_lum = sky.light_recover(dec_field,thr,bkg_mean,(ker_sigma,Dker_sigma),extraction=light_results,binning=BINNING,results=PLOTTING,display_fig=PLOTTING)

# mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()           #: mean value
# # average and compute the STD
# mean_rec = np.mean(rec_lum)
# observable = STARS.lum>bkg_mean
# # print
# print(f'Slum {STARS.lum[:4]}')
# print(f'Rlum {rec_lum[:4]}')
# print(f'S: {mean_lum:.2e}\tL: {mean_rec:.2e}\t{(mean_rec-mean_lum)/mean_lum:.2%}')
# print(f'FOUND: {len(rec_lum)}')
# print(f'EXPEC: {len(STARS.lum[observable])}')
# # sky.store_results('source',[STARS.m,STARS.lum,STARS.pos[0],STARS.pos[1]],main_dir=DIR_NAME,columns=['M','L','X','Y'])
# # sky.store_results('recovered',[rec_lum,Drec_lum,light_results['pos'][0],light_results['pos'][1]],main_dir=DIR_NAME,columns=['L','DL','X','Y'])

# if PLOTTING:
#     plt.figure()
#     plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
#     plt.hist(rec_lum,BINNING,histtype='step',color='blue')
#     plt.axvline(mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed',alpha=0.5)
#     # plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
#     plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed',alpha=0.5)
#     # plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dashed')
#     plt.legend(fontsize=FONTSIZE)
#     plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
#     plt.ylabel('counts',fontsize=FONTSIZE)
#     plt.grid(linestyle='dashed',color='gray',alpha=0.5)
#     plt.show()

### NO BKG
# BKG_PARAM = ('Uniform',(0,0))
# # built the field
# STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(quantize=DC,det_param=DET_PARAM,back_param=BKG_PARAM,results=PLOTTING,display_fig=False)

# # science_frame  = master_light - master_dark
# # Dscience_frame = np.sqrt(Dmaster_light**2 + Dmaster_dark**2)
# science_frame  = master_light
# Dscience_frame = Dmaster_light

# print(STARS.mean_lum(),STARS.lum.max(),STARS.lum.min())

# print('NEGATIVE',len(np.where(master_dark <0)[0]) )
# print('NEGATIVE',len(np.where(science_frame <0)[0]) )

# print(np.var(master_light))
# plt.figure(1)
# plt.hist(Dmaster_light.flatten(),1000,histtype='step')
# plt.figure(2)
# cnts, bins,_ = plt.hist(master_light.flatten(),500,histtype='step')
# plt.xscale('log')
# plt.yscale('log')
# plt.figure(3)
# cnts, bins,_ = plt.hist(science_frame.flatten(),500,histtype='step')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# ### RESTORATION
# ## Background Estimation
# bkg_mean, bkg_sigma = sky.bkg_est(science_frame,display_plot=PLOTTING)
# print('DC',DC)
# ## Kernel Estimation
# thr = bkg_mean + bkg_sigma  #: the threshold for searching algorithm
# max_num_obj = 10            #: number of objects at the most 
# obj_param = [[],[],[],[]]   #: list of gaussian fit results for each obj
# # extract objects
# objs, Dobjs, _ = sky.searching(science_frame,thr,0,Dscience_frame,ker_sigma=None,num_objs=max_num_obj,cntrl=20,log=True,obj_params=obj_param,display_fig=PLOTTING)
# # fit a 2DGaussian profile to extraction
# ker_sigma, Dker_sigma = sky.kernel_estimation(objs,Dobjs,(bkg_mean,0),obj_param=obj_param,results=PLOTTING,title='title-only')
# # compute the kernel
# atm_kernel = sky.Gaussian(sigma=ker_sigma)

# ## RL Algorithm
# dec_field = sky.LR_deconvolution(science_frame,atm_kernel,np.zeros(science_frame.shape),bkg_mean,bkg_sigma,max_r=2000,results=PLOTTING,display_fig=True,norm='log')

# if PLOTTING:
#     fig,(ax1,ax2) = plt.subplots(1,2)
#     ax1.set_title('Before deconvolution',fontsize=FONTSIZE+2)
#     sky.field_image(fig,ax1,science_frame)
#     ax2.set_title('After deconvolution',fontsize=FONTSIZE+2)
#     sky.field_image(fig,ax2,dec_field)
#     plt.show()


# ## Light Recover
# light_results = {}
# rec_lum, Drec_lum = sky.light_recover(dec_field,thr,bkg_mean,(ker_sigma,Dker_sigma),extraction=light_results,binning=BINNING,results=PLOTTING,display_fig=PLOTTING)

# mean_lum = STARS.mean_lum() if STARS.mean_lum() is not None else STARS.lum.mean()           #: mean value
# # average and compute the STD
# mean_rec = np.mean(rec_lum)
# observable = STARS.lum>bkg_mean
# # print
# print(f'Slum {STARS.lum[:4]}')
# print(f'Rlum {rec_lum[:4]}')
# print(f'S: {mean_lum:.2e}\tL: {mean_rec:.2e}\t{(mean_rec-mean_lum)/mean_lum:.2%}')
# print(f'FOUND: {len(rec_lum)}')
# print(f'EXPEC: {len(STARS.lum[observable])}')
# # sky.store_results('source',[STARS.m,STARS.lum,STARS.pos[0],STARS.pos[1]],main_dir=DIR_NAME,columns=['M','L','X','Y'])
# # sky.store_results('recovered',[rec_lum,Drec_lum,light_results['pos'][0],light_results['pos'][1]],main_dir=DIR_NAME,columns=['L','DL','X','Y'])

# if PLOTTING:
#     plt.figure()
#     plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
#     plt.hist(rec_lum,BINNING,histtype='step',color='blue')
#     plt.axvline(mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed',alpha=0.5)
#     # plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
#     plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed',alpha=0.5)
#     # plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dashed')
#     plt.legend(fontsize=FONTSIZE)
#     plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
#     plt.ylabel('counts',fontsize=FONTSIZE)
#     plt.grid(linestyle='dashed',color='gray',alpha=0.5)
#     plt.show()
