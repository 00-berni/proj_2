import numpy as np
import matplotlib.pyplot as plt
from skysimulation import field
from skysimulation import display
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

K = field.K

def initialize(dim: int, num: int, display_fig: bool = False, **kwargs):
    beta = field.BETA
    masses = np.linspace(field.MIN_m,8, num)
    lums = masses**beta
    F = np.zeros((dim,dim))
    ynum = 5
    xnum = num // ynum + num % ynum
    print(xnum,ynum)
    x = np.array([i*dim // (xnum+1) for i in range(1,xnum+1)]*ynum)
    y = np.sort(np.array([(i+1)*dim // (ynum+1) for i in range(ynum)]*xnum))
    print(len(x),len(y))
    if num != xnum*ynum: 
        diff = xnum*ynum - num
        lums = np.append(lums,[0]*diff)
    F[y,x] = lums * K

    if display_fig:
        if 'title' not in kwargs:
            kwargs['title'] = f'Inizialized Field\nMass range [{masses.min():.1f}, {masses.max():.1f}]'
            field.fast_image(F,**kwargs)
    return F, masses, (x,y)    


def add_effects(F,masses,coor,back,atm,det,**kwargs):
    N = len(F)
    if 'results' not in kwargs: kwargs['results'] = False
    if 'figure' not in kwargs: kwargs['figure'] = False
    if 'norm' not in kwargs: kwargs['norm'] = 'log'
    if 'v' not in kwargs: kwargs['v'] = 0
    results = kwargs['results']
    figure = kwargs['figure']
    norm = kwargs['norm']
    v = kwargs['v']
    print('\nBackground')
    Fb = F + field.noise(back,dim=N,infos=True,display_fig=figure,norm=norm,v=v,title='Background')
    if figure:
        field.fast_image(Fb,v=v,norm=norm,title='F + B')
    print('\nSeeing')
    Fs = field.atm_seeing(F,atm,display_fig=figure,title='Bare Atm Seeing',norm=norm,v=v)
    Fbs = field.atm_seeing(Fb,atm,figure,norm=norm,v=v)
    print('\nNoise')
    Fbsn = Fbs + field.noise(det,dim=N,infos=True,display_fig=figure,title='Noise',norm=norm,v=v)
    if figure:
        field.fast_image(Fbsn,v=0,norm='linear',title='Field')
    if results:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,15))
        plt.subplots_adjust(wspace=0.01)
        fig.suptitle('Initialization Process',fontsize=20)

        ax1.set_title('Initial Field',fontsize=14)
        display.field_image(fig,ax1,F,v=2,norm=norm)
        x,y = coor
        for i in range(len(masses)):
            ax1.annotate(f'{masses[i]:.1f}',(x[i],y[i]),(x[i]-4,y[i]-3),fontsize=12)

        ax2.set_title('Field and Background',fontsize=14)
        display.field_image(fig,ax2,Fb,v=v,norm=norm)
        
        ax3.set_title('Seeing',fontsize=14)
        display.field_image(fig,ax3,Fbs,v=v,norm=norm)
        
        ax4.set_title('Field and Noise',fontsize=14)
        display.field_image(fig,ax4,Fbsn,v=v,norm=norm)
        
        plt.show()


    return Fbsn


if __name__ == '__main__':
    N = 100
    M = 20
    figure = False
    norm = 'log'
    v = 0
    print('Initializing')
    F, masses, coor = initialize(N,M,display_fig=figure,v=1,norm=norm)
    
    print('\nI RUN')
    back = field.BACK_PARAM
    # back = ('Gaussian',(field.BACK_MEAN*1e1,field.BACK_MEAN*1e1*20e-2))
    det = field.NOISE_PARAM
    atm = field.SEEING_SIGMA

    iter = 2
    results = []
    nn = []
    for _ in range(iter):
        I = add_effects(F,masses,coor,back,atm,det,figure=figure,norm=norm,v=v,results=False)
        results += [I]
        n = restore.bkg_est(I,True)
        print(n)
        print(n/K)
        nn += [n]
    
    n0 = sum(nn)/len(nn)
    nm = (max(nn)+min(nn))/2
    print('n0 ',n0/K)
    print('nm ',nm/K)
    print('min',min(nn)/K)
    # print('\nII RUN')
    # back = ('Gaussian',(0.2, field.BACK_SIGMA))
    # det = ('Gaussian',(field.NOISE_MEAN,field.NOISE_SIGMA))
    # atm = field.SEEING_SIGMA
    # add_effects(F,back,atm,det,figure=figure,norm=norm,v=v)

