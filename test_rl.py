import numpy as np
import matplotlib.pyplot as plt
from skysimulation import field
import skysimulation.restoration as restore


# def field_maker(num_obj: int, dim: int = field.N, beta: float = field.BETA, back: float = field.BACK, noise: float = field.NOISE, sigma: float = field.SIGMA):
#     masses = np.linspace(field.MIN_m, field.MAX_m,num_obj)
#     print('masses:\t',masses)
#     lums = masses**beta
#     print('brightnesses:\t',lums)
#     F = np.zeros((dim,dim))
#     for i in range(num_obj):
#         print(f'{masses[i]:.2f} Msol -> ({x[i]:02d},{y[i]:02d})')
#     field.fast_image(F, v=1)
#     n_b = field.noise(field.Uniform(back),dim)
#     Ff = F + n_b
#     Ff = field.atm_seeing(Ff,sigma)
#     Ff = Ff + field.noise(field.Uniform(noise),dim)
#     field.fast_image(Ff, v=1)
#     return F, Ff

def initialize(dim: int, num: int, display_fig: bool = False, **kwargs):
    beta = field.BETA
    masses = np.linspace(field.MIN_m,field.MAX_m, num)
    lums = masses**beta
    F = np.zeros((dim,dim))
    ynum = 3
    xnum = num // ynum + num % ynum
    print(xnum,ynum)
    x = np.array([i*dim // (xnum+1) for i in range(1,xnum+1)]*ynum)
    y = np.sort(np.array([(i+1)*dim // (ynum+1) for i in range(ynum)]*xnum))
    print(len(x),len(y))
    if num != xnum*ynum: 
        diff = xnum*ynum - num
        lums = np.append(lums,[0]*diff)
    F[y,x] = lums

    if display_fig:
        if 'title' not in kwargs:
            kwargs['title'] = f'Inizialized Field\nMass range [{masses.min():.1f}, {masses.max():.1f}]'
            field.fast_image(F,**kwargs)
    return F    


def add_effects(F,back,atm,det,**kwargs):
    N = len(F)
    figure = kwargs['figure']
    norm = kwargs['norm']
    v = kwargs['v']
    print('\nBackground')
    Fb = F + field.noise(back,N,True,figure,norm=norm,v=v,title='Background')
    field.fast_image(Fb,v=v,norm=norm,title='F + B')
    print('\nSeeing')
    Fs = field.atm_seeing(F,atm,figure,title='Bare Atm Seeing',norm=norm,v=v)
    Fbs = field.atm_seeing(Fb,atm,figure,norm=norm,v=v)
    print('\nNoise')
    Fbsn = Fbs + field.noise(det,N,True,figure,title='Noise',norm=norm,v=v)
    
    field.fast_image(Fbsn,v=0,norm='linear',title='Field')

    return Fbsn


if __name__ == '__main__':
    N = 100
    M = 15
    figure = True
    norm = 'linear'
    v = 1
    print('Initializing')
    F = initialize(N,M,display_fig=figure,v=v,norm=norm)
    
    print('\nI RUN')
    back = field.BACK_PARAM
    det = field.NOISE_PARAM
    atm = field.SEEING_SIGMA
    I = add_effects(F,back,atm,det,figure=figure,norm=norm,v=v)

    # print('\nII RUN')
    # back = ('Gaussian',(0.2, field.BACK_SIGMA))
    # det = ('Gaussian',(field.NOISE_MEAN,field.NOISE_SIGMA))
    # atm = field.SEEING_SIGMA
    # add_effects(F,back,atm,det,figure=figure,norm=norm,v=v)

