import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky

FONTSIZE = 18

# MAIN_DIR = 'default'

# source = sky.Star.load_data('source',MAIN_DIR)
# recover = sky.open_data('recovered',MAIN_DIR,'array')

# mean_lum0 = source.mean_lum()


# bkg_mean = np.linspace(3.5,5.0,10)

# mean_s = np.loadtxt(sky.os.path.join(sky.RESULT_DIR,'source.txt'),unpack=True)[-1]
# mean_nart = [] 
# mean_wart = []
# for bkg in bkg_mean:
#     path = sky.os.path.join(sky.RESULT_DIR,f'bkg-{bkg:.2f}.txt')
    
MAIN_DIR = 'multi-bkg-real'
MAIN_DIR = 'multi-new-corr'
BKG_VALUES = np.array([3.2,4.2,5.2,6.2,7.2,8.2,9.2,10,11,15,20])
BKG_ITER = 20
# FILES_NAME =

source = sky.Star.load_data(main_dir='default')

mean_sl = source.mean_lum()
print(mean_sl)

# (bkg, 20, [L, DL, X, Y],numpoints)
mean_lightdata = np.array([[ np.mean(sky.open_data(f'bkg-{bkg:.2f}_{i:02d}',main_dir=MAIN_DIR,out_type='array')[0]) for i in range(BKG_ITER)] for bkg in BKG_VALUES])
Dmean_lightdata = np.array([[ np.std(sky.open_data(f'bkg-{bkg:.2f}_{i:02d}',main_dir=MAIN_DIR,out_type='array')[0],ddof=1) for i in range(BKG_ITER)] for bkg in BKG_VALUES])

print(np.mean(mean_lightdata,axis=1))

# for bkg in BKG_VALUES:
#     for i in range(BKG_ITER)
#         data 
# (bkg, 20)
mean_rl, Dmean_rl = sky.mean_n_std(mean_lightdata,axis=1)

# mean_rl = np.mean(mean_lightdata,axis=1)
diff = mean_rl - mean_sl
for df, dm in zip(diff,Dmean_rl):
    print(df,dm,dm/df*100,'%')

select = [0,1,4,-3,-2,-1] #slice(-5,None)

plt.figure()
plt.errorbar(BKG_VALUES[select],diff[select],fmt='.--')
plt.grid(linestyle='dotted',alpha=0.7)
plt.xlabel('$\\bar{n}_B$',fontsize=FONTSIZE)
plt.ylabel('$\\langle\\ell\\rangle - \\langle\\ell_0\\rangle$',fontsize=FONTSIZE)
plt.figure()
plt.errorbar(BKG_VALUES,diff,fmt='.--')
for i in range(BKG_ITER):
    plt.errorbar(BKG_VALUES,mean_lightdata[:,i]-mean_sl,fmt='.')
plt.show()


### POISSON DETECTOR
# BINNING = 20
# MAIN_DIR = 'poisson_det'

# source = sky.Star.load_data(main_dir=MAIN_DIR)
# rec_lum, Drec_lum, xpos, ypos = sky.open_data('recovered',main_dir=MAIN_DIR,out_type='array')
# xpos, ypos = xpos.astype('int'), ypos.astype('int')
# mean_lum = source.mean_lum()
# mean_rec = np.mean(rec_lum)

# plt.figure()
# plt.title('Restored distribution in brightness',fontsize=FONTSIZE+2)
# plt.hist(rec_lum,BINNING,histtype='step',color='blue')
# plt.hist(source.lum,40,histtype='step',color='red')
# plt.axvline(mean_rec,0,1,color='blue',label='mean recovered brightness',linestyle='dashed',alpha=0.5)
# # plt.axvspan(mean_rec-Dmean_rec,mean_rec+Dmean_rec,facecolor='blue',alpha=0.4)
# plt.axvline(mean_lum,0,1,color='red',label='mean source brightness',linestyle='dashed',alpha=0.5)
# # plt.axvline(bkg_mean,0,1,color='green',label='mean background',linestyle='dashed')
# plt.legend(fontsize=FONTSIZE)
# plt.xlabel('$\\ell$ [a.u.]',fontsize=FONTSIZE)
# plt.ylabel('counts',fontsize=FONTSIZE)
# plt.grid(linestyle='dashed',color='gray',alpha=0.5)
# plt.show()
