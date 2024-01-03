import numpy as np
import matplotlib.pyplot as plt
from skysimulation import field
from skysimulation import display
import skysimulation.restoration as restore


K = field.K

def initialize(dim: int, num: int, display_fig: bool = False, **kwargs):
    beta = field.BETA
    masses = np.linspace(field.MIN_m,5, num)
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
    norm = 'linear'
    v = 0
    print('Initializing')
    F, masses, coor = initialize(N,M,display_fig=figure,v=1,norm=norm)
    
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
    
    # diff1 = I[1:]/I[:-1]
    # diff2 = I[:,1:]/I[:,:-1]
    # diff = np.append(diff1,diff2) - 1
    Iflat = I.flatten()
    # diff = np.array([ Iflat[i]/np.delete(Iflat,i) for i in range(len(Iflat)) ]).flatten() - 1
    diff = np.array([i/I for i in I])
    print(f'diff = {diff.shape}')
    print(f'diff -> {diff[0]}')
    diff = diff.flatten()
    print(f'diff = {diff.shape}')
    pos = np.where(diff!=1.)[0]
    print(len(np.where(diff==1.)[0]))
    diff = diff[pos] -1
    print(f'diff = {diff.shape}')
    bins = np.linspace(min(diff),max(diff),np.sqrt(len(diff)).astype(int))
    counts, bins = np.histogram(diff,bins=bins)
    maxpos = counts.argmax()
    maxval = (bins[maxpos+1] + bins[maxpos])/2
    from scipy import stats
    # x = bins[:-1]
    # y = counts
    def gauss_fit(x,*args):
        k,mu,sigma = args
        r = (x-mu)/sigma
        return k * np.exp(-r**2/2)
    # from scipy.optimize import curve_fit
    # cut = np.where(counts >= 500)[0]
    # edges = (bins[cut].min(), bins[cut].max())
    # cut = np.where(np.logical_and(edges[0] <= diff , diff <= edges[1]))[0]
    thr = counts[maxpos] * 20e-2
    edges = np.where(counts >= thr)[0][[0,-1]]
    print('edges',edges)
    edges = bins[edges]
    print('edges',edges)
    dlim = abs(edges[0]-diff).argmin()
    ulim = abs(edges[1]-diff).argmin()
    dlim, ulim = min(dlim,ulim), max(dlim,ulim)
    print('lim',dlim,ulim)
    cut = slice(dlim,ulim)
    (mu, sigma) = stats.norm.fit(diff[cut],loc=maxval,scale=1)
    khist = counts.sum()*np.mean(bins[1:]-bins[:-1])
    k = khist / np.sqrt(2*np.pi) / sigma
    # initial_values = [max(y),maxval,0.5]
    # pop, pcov = curve_fit(gauss_fit,x,y,initial_values)
    # k,mu,sigma = pop
    print('maxval',maxval)
    print('mu',mu)
    print('sigma',sigma)
    pop = [k,mu,sigma]
    plt.figure()
    plt.stairs(counts,bins,fill=True)
    xx = np.linspace(min(bins),max(bins),1000)
    plt.plot(xx,gauss_fit(xx,*pop),color='orange')
    plt.axhline(thr,0,1,color='black',linestyle='dotted')
    plt.axvline(maxval,0,1,linestyle='--',color='red')
    plt.axvline(mu,0,1,linestyle='--',color='violet')
    plt.show()

    err = (mu+1)*np.mean(I)
    print(f'!err={err}')
    err = (mu+1)*max(bkg,dark.mean())#np.mean(I)
    print(f'!err={err}')
    print(max(bkg,dark.mean()))
    raise

    objs = restore.object_isolation(I,max(bkg,dark.mean()),size=7,objnum=20,reshape=True,reshape_corr=True,sel_cond=True,display_fig=False,norm=norm)

    if objs is not None:
        obj = objs[1]
        sigma = restore.kernel_fit(obj,err,True)

    # print('\nII RUN')
    # back = ('Gaussian',(0.2, field.BACK_SIGMA))
    # det = ('Gaussian',(field.NOISE_MEAN,field.NOISE_SIGMA))
    # atm = field.SEEING_SIGMA
    # add_effects(F,back,atm,det,figure=figure,norm=norm,v=v)

