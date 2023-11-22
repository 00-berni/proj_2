import numpy as np
import matplotlib.pyplot as plt
from skysimulation import field
import skysimulation.restoration as restore


def field_maker(num_obj: int, dim: int = field.N, beta: float = field.BETA, back: float = field.BACK, noise: float = field.NOISE, sigma: float = field.SIGMA):
    masses = np.linspace(field.MIN_m, field.MAX_m,num_obj)
    print('masses:\t',masses)
    lums = masses**beta
    print('brightnesses:\t',lums)
    F = np.zeros((dim,dim))
    ynum = 2
    xnum = num_obj // ynum + num_obj % ynum
    x = np.array([i*dim // (xnum+1) for i in range(1,xnum+1)]*ynum)
    y = np.sort(np.array([(i+2)*dim // 5 for i in range(ynum)]*xnum))
    print(x,y)
    if num_obj != xnum*ynum: 
        x = x[:-1]
        y = y[:-1]
    F[y,x] = lums
    for i in range(num_obj):
        print(f'{masses[i]:.2f} Msol -> ({x[i]:02d},{y[i]:02d})')
    field.fast_image(F, v=1)
    n_b = field.noise(field.Uniform(back),dim)
    Ff = F + n_b
    Ff = field.atm_seeing(Ff,sigma)
    Ff = Ff + field.noise(field.Uniform(noise),dim)
    field.fast_image(Ff, v=1)
    return F, Ff




if __name__ == '__main__':
    num_obj = 10
    F0, F = field_maker(num_obj)
    dark = restore.dark_elaboration(field.Uniform(field.NOISE)).max()
    back = restore.bkg_est(F-dark)
    print(f'Extimated background maxval:\t{10**back}')
    back = 10**back
    extr = restore.object_isolation(F,max(back,dark.max()),objnum=num_obj//2,reshape=True,reshape_corr=True)
    ext_kernel = restore.kernel_extimation(extr,back,dark,len(F),True)
    F_ext = restore.LR_deconvolution(F,ext_kernel,back,dark)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(F,norm='log')
    plt.colorbar()
    plt.subplot(1 ,2,2)
    plt.imshow(F_ext,norm='log')
    plt.colorbar()
    plt.show()

