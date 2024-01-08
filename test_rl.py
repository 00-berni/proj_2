import numpy as np
import matplotlib.pyplot as plt
from skysimulation import field
from skysimulation import display
import skysimulation.restoration as restore


K = field.K

def initialize(dim: int, num: int,max_mass: float | int = 8, display_fig: bool = False, **kwargs):
    beta = field.BETA
    masses = np.linspace(field.MIN_m,max_mass,num)
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
    # F[5,-1] = 8.6**beta * K
    # F[5,-9] = 8.5**beta * K
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
        display.field_image(fig,ax1,F,v=2,norm='log')
        x,y = coor
        for i in range(len(masses)):
            ax1.annotate(f'{masses[i]:.1f}',(x[i],y[i]),(x[i]-4,y[i]+3),fontsize=12)

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
    norm = 'linear'
    v = 0
    print('Initializing')
    max_mass = 5
    F, masses, coor = initialize(N,M,max_mass,display_fig=figure,v=1,norm=norm)
    
    print('\nI RUN')
    back = field.BACK_PARAM
    det = field.NOISE_PARAM
    atm = field.SEEING_SIGMA

    iter = 1
    results = []
    nn = []
    for _ in range(iter):
        I = add_effects(F,masses,coor,back,atm,det,figure=figure,norm=norm,v=v,results=True)
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

    dark = restore.dark_elaboration(det)
    bkg = nn[0]
    I = results[0]
    
    mean_val = max(bkg,dark.mean())
    objs, obj_pos = restore.object_isolation(I,mean_val,size=7,objnum=20,reshape=True,reshape_corr=True,sel_cond=True,display_fig=False,norm=norm)

    if objs is not None:
        try:
            err = restore.err_estimation(I,mean_val,display_plot=True)
        except:
            err = None
        kernel,(sigma, Dsigma) = restore.kernel_estimation(objs,err,N,all_results=False,display_plot=False)

        rec_I = restore.LR_deconvolution(I,kernel,mean_val,iter=50,sel='rl',display_fig=True)
        dim = len(rec_I)
        size = 0
        cond = np.array([False]*4)
        for i in range(dim-1):
            diff1 = rec_I[i+1,i+1] - rec_I[i,i]
            diff2 = rec_I[dim-1-i-1,dim-1-i-1] - rec_I[dim-1-i,dim-1-i]
            diff3 = rec_I[i+1,dim-1-i-1] - rec_I[i,dim-1-i]
            diff4 = rec_I[dim-1-i-1,i+1] - rec_I[dim-1-i,i]
            if diff1 <= 0:
                if i > size: size = i
                cond[0] = True
            if diff2 <= 0:
                if i > size: size = i
                cond[1] = True
            if diff3 <= 0:
                if i > size: size = i
                cond[2] = True
            if diff4 <= 0:
                if i > size: size = i
                cond[3] = True
            if all(cond): break
        print('size',size)
        size0 = size
        for i in range(size+1,dim-1):
            diff1 = rec_I[:,i+1] - rec_I[:,i]
            diff2 = rec_I[:,dim-1-i-1] - rec_I[:,dim-1-i]
            diff3 = rec_I[i+1,:] - rec_I[i,:]
            diff4 = rec_I[dim-1-i-1,:] - rec_I[dim-1-i,:]
            diff1 = diff1.max()
            diff2 = diff2.max()
            diff3 = diff3.max()
            diff4 = diff4.max()
            if max(diff1,diff2,diff3,diff4) >= 0:
                size = i
                break
        print('size',size)
        fig, (ax1,ax2) = plt.subplots(1,2)
        display.field_image(fig,ax1,I)
        ax1.axhline(size0,0,1,color='red')
        ax1.axhline(size0+size0//3,0,1,color='orange')
        ax1.axhline(size,0,1)
        ax1.axhline(dim-1-size,0,1)
        ax1.axvline(size,0,1)
        ax1.axvline(dim-1-size,0,1)
        display.field_image(fig,ax2,rec_I)
        ax2.axhline(size0,0,1,color='red')
        ax2.axhline(size0+size0//3,0,1,color='orange')
        ax2.axhline(size,0,1)
        ax2.axhline(dim-1-size,0,1)
        ax2.axvline(size,0,1)
        ax2.axvline(dim-1-size,0,1)
        plt.show()
        display.fast_image(rec_I-I)
    else:
        print('[ALERT] - It is not possible to recover the field!\nTry to change parameters')
