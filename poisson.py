import numpy as np
import matplotlib.pyplot as plt
import skysimulation as sky


### INITIALIZATION
## Parameters
RESOLUTION = sky.MIN_m**sky.BETA*sky.K * 1e-2
LAMBDA = 1e9
# LAMBDA = sky.BACK_MEAN*sky.K * 50
K = 1
DET_PARAM = ('Poisson',(LAMBDA,K))
PLOTTING = True

# built the field
STARS, (master_light, Dmaster_light), (master_dark, Dmaster_dark) = sky.field_builder(quantize=RESOLUTION,det_param=DET_PARAM,results=PLOTTING,display_fig=False)

science_frame = master_light - master_dark

print(STARS.mean_lum(),STARS.lum.max(),STARS.lum.min())

sky.fast_image(science_frame)
print('NEGATIVE',len(np.where(master_dark <0)[0]) )
print('NEGATIVE',len(np.where(science_frame <0)[0]) )

print(np.var(master_light))
plt.figure(1)
plt.hist(Dmaster_light.flatten(),1000,histtype='step')
plt.xscale('log')
plt.figure(2)
cnts, bins,_ = plt.hist(master_light.flatten(),1000,histtype='step')
plt.show()

### RESTORATION
bkg_mean, bkg_sigma = sky.bkg_est(science_frame,display_plot=PLOTTING)
print(sky.BACK_MEAN)
