import numpy as np
import matplotlib.pyplot as plt
import skysimulation.field as fld
import skysimulation.restoration as rst
from skysimulation.display import fast_image, field_image

### MAIN ###
if __name__ == '__main__':
    sigma = 3
    kernel = fld.Gaussian(sigma).kernel()
    centre = len(kernel)//2
    obj0 = np.zeros(kernel.shape)
    obj0[centre, centre] = 1
    m_bkg = 1e-3
    bkg_distr = fld.Gaussian(m_bkg*20e-2, m_bkg)
    bkg = bkg_distr.field(obj0.shape)
    if len(np.where(bkg<0)[0]) != 0:
        bkg = np.sqrt(bkg**2)
    obj0 = fld.field_convolve(obj0,kernel,bkg_distr)
    obj1 = obj0 + bkg
    obj2 = obj0 + bkg*10
    obj3 = obj0 + bkg*50
    obj4 = obj0 + bkg*80
    noise = bkg*100

    objs  = [obj1, obj2, obj3, obj4, noise]
    names = ['Luminous Obj','Worse Obj','Noisy Obj','Noise Obj','Noise']
    lv = []
    for obj, name, k in zip(objs,names,[1,10,50,80,100]):
        # fast_image(obj,name + ' before convolution')
        # obj = fld.field_convolve(obj,kernel,bkg_distr)
        xdim, ydim = obj.shape
        xmax, ymax = rst.peak_pos(obj)
        maxval = obj[xmax,ymax]
        est_bkg = m_bkg*k / maxval*100
        dim = max([xmax, ymax, xdim-xmax, ydim-ymax])
        mean_obj = np.array([[100],[100]])
        for i in range(1,dim):
            xl = xmax - i
            xr = xdim - xmax - i
            yl = ymax - i
            yr = ydim - ymax - i
            mean_val_1 = []
            mean_val_2 = []
            if xl >= 0: mean_val_1 += [obj[xl,ymax]]
            if xr >  0: mean_val_1 += [obj[xr,ymax]]
            if yl >= 0: mean_val_1 += [obj[xmax,yl]]
            if yr >  0: mean_val_1 += [obj[xmax,yr]]
            if xl >= 0 and yl >= 0: mean_val_2 += [obj[xl,yl]]
            if xr >  0 and yl >= 0: mean_val_2 += [obj[xr,yl]]
            if xl >= 0 and yr >  0: mean_val_2 += [obj[xl,yr]]
            if xr >  0 and yr >  0: mean_val_2 += [obj[xr,yr]]
            if len(mean_val_1) != 0 and len(mean_val_2) != 0:
                mean_val_1 = np.mean(mean_val_1) / maxval*100
                mean_val_2 = np.mean(mean_val_2) / maxval*100
                mean_obj = np.append(mean_obj, [[mean_val_1], [mean_val_2]], axis=1)
        grad1 = np.diff(mean_obj[0])
        grad2 = np.diff(mean_obj[1])
        gg1 = np.diff(grad1)
        gg2 = np.diff(grad2)
        print('mean_obj', mean_obj)
        print('grad1',grad1)
        print('grad2',grad2)
        print('gg1',gg1)
        print('gg2',gg2)        

        hm_pos = np.array([abs(mobj-50).argmin() for mobj in mean_obj ], dtype=int)

        in_mean = np.array([ mobj[:pos+1].mean() for mobj, pos in zip(mean_obj,hm_pos)])
        out_mean = np.array([ mobj[pos+1:].mean() for mobj, pos in zip(mean_obj,hm_pos) if pos+1 < len(mobj)])
        print('in',in_mean)
        print('out',out_mean)
        sci_obj = mean_obj - est_bkg
        mm = np.mean(sci_obj[:,1:],axis=1)
        plt.figure()
        plt.subplot(2,1,1)
        plt.title(f'gg1 = {gg1.mean():.2}, gg2 = {gg2.mean():.2} ; mm = {mm.mean():.2f}')
        plt.plot(mean_obj[0],'.--')
        plt.plot(gg1,'+--')
        plt.axhline(0,0,1,color='black')
        plt.axhline(mm[0],0,1,color='pink',linestyle='dotted')
        plt.axvline(hm_pos[0],0,1,color='green',linestyle='dashed')
        plt.subplot(2,1,2)
        plt.plot(mean_obj[1],'.--')
        plt.plot(gg2,'+--')
        plt.axhline(0,0,1,color='black')
        plt.axhline(mm[1],0,1,color='pink',linestyle='dotted')
        plt.axvline(hm_pos[1],0,1,color='green',linestyle='dashed')
        fast_image(obj)        
        lv += [mm.mean()]

    print('Levels', lv)