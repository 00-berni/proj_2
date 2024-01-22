import numpy as np
import matplotlib.pyplot as plt
from skysimulation import NDArray
from skysimulation import field
from skysimulation import display
import skysimulation.restoration as restore


K = field.K
st_pos = 20

def initialize(dim: int, num: int, max_mass: float | int = 8, pos: str = 'grid', set_pos: NDArray | None = None, display_fig: bool = False, **kwargs):
    beta = field.BETA
    masses = np.linspace(field.MIN_m,max_mass,num)
    lums = masses**beta
    F = np.zeros((dim,dim))
    if pos == 'grid' or (pos == 'set' and set_pos is None):
        ynum = 5
        xnum = num // ynum + num % ynum
        print(xnum,ynum)
        x = np.array([i*dim // (xnum+1) for i in range(1,xnum+1)]*ynum)
        y = np.sort(np.array([(i+1)*dim // (ynum+1) for i in range(ynum)]*xnum))
        print(len(x),len(y))
        if num != xnum*ynum: 
            diff = xnum*ynum - num
            lums = np.append(lums,[0]*diff)
    elif pos == 'random':
        y,x = field.star_location(num,dim,overlap=False)
    elif pos == 'set':
        y,x = set_pos
    F[y,x] = lums * K
    # F[st_pos,st_pos] = 4**beta * K
    # F[5,-1] = 8.6**beta * K
    # F[5,-9] = 8.5**beta * K
    if display_fig:
        if 'title' not in kwargs:
            kwargs['title'] = f'Inizialized Field\nMass range [{masses.min():.1f}, {masses.max():.1f}]'
            field.fast_image(F,**kwargs)
    return F, masses, (y,x)    


def add_effects(F,masses,coor,back,atm,det,pos_mode: str = 'grid',**kwargs):
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
        if pos_mode == 'grid':
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
    figure = True
    norm = 'linear'
    v = 0
    pos_mode = 'random'
    set_pos = None
    # pos_mode = 'set'
    # set_pos = np.array([ [0]*10 + [N-1]*10, [i*4 for i in range(20)] ])
    print('Initializing')
    max_mass = 5
    F, masses, coor = initialize(N,M,max_mass,pos=pos_mode,set_pos=set_pos,display_fig=figure,v=1,norm=norm)

    print('\nI RUN')
    back = field.BACK_PARAM
    det = field.NOISE_PARAM
    atm = field.SEEING_SIGMA

    iter = 1
    results = []
    nn = []
    for _ in range(iter):
        I = add_effects(F,masses,coor,back,atm,det,pos_mode=pos_mode,figure=figure,norm=norm,v=v,results=True)
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

    # dark = restore.dark_elaboration(det)
    # bkg = nn[0]
    # I = results[0]
    
    # mean_val = max(bkg,dark.mean())
    # objs, obj_pos = restore.object_isolation(I,mean_val,size=7,objnum=20,reshape=True,reshape_corr=True,sel_cond=True,display_fig=False,norm=norm)

    # if objs is not None:
    #     try:
    #         err = restore.err_estimation(I,mean_val,display_plot=True)
    #     except:
    #         err = None
    #     kernel,(sigma, Dsigma) = restore.kernel_estimation(objs,err,N,all_results=False,display_plot=False)

    #     rec_I = restore.LR_deconvolution(I,kernel,mean_val,iter=50,sel='rl',display_fig=True)
    #     mask = restore.mask_filter(rec_I,I,True)
    #     lum, pos = restore.find_objects(rec_I,I,kernel,mean_val,sel_pos=obj_pos,display_fig=False)
    #     l, Dl = lum - mean_val #- dark.mean()
        
    #     index  = pos.copy()
    #     index0 = np.array(coor)[:,::-1]

    #     min_dim = min(len(index[0]),len(index0[0]))
    #     dx, dy = abs(index[:,:min_dim] - index0[:,:min_dim])
    #     discr = 2
    #     mat_pos = np.where(np.logical_and(dx<=discr,dy<=discr))[0]        
    #     matches = len(mat_pos)

    #     print('\n----------------\n')
    #     print(f'MAX MASS - {max_mass}')
    #     print(f'Found:\t{len(l)} / {M}')
    #     print(f'Precision:\t{len(l)/M*100:.2f} %')
    #     print(f'Match:\t{matches} / {M}')
    #     print(f'Accuracy:\t{matches/len(l)*100:.2f} %')


    #     # print('Center Value',I[st_pos,st_pos],rec_I[st_pos,st_pos])
    #     # print('MAXes',l0.max(),l.max())
    #     # print('MINes',l0.min(),l.min())
    #     # fig, ax = plt.subplots(1,1)
    #     # bins = np.linspace(min(l.min(),l0.min()),max(l.max(),l0.max()),len(l0)*4)
    #     # l_counts, bins = np.histogram(l,bins=bins)
    #     # l0_counts, bins = np.histogram(l0,bins=bins)
    #     # ax.stairs(l_counts,bins,label='estimated')
    #     # ax.stairs(l0_counts,bins,linestyle='dashed',label='original')
    #     # lm = np.linspace(masses.min()**field.BETA,masses.max()**field.BETA,6) *K
    #     # ax.set_xticks(lm)
    #     # ax.set_xticklabels([f'{(i/K)**(1/field.BETA):.1f}' for i in lm])
    #     # # ax.set_xscale('log')
    #     # ax.legend()
    #     # plt.show()
    # else:
    #     print('[ALERT] - It is not possible to recover the field!\nTry to change parameters')
