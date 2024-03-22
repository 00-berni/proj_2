import numpy as np
import matplotlib.pyplot as plt
import skysimulation.display as dpl
import skysimulation.field as fld
import skysimulation.restoration as rst





if __name__ == '__main__':

    acq_num = 3
    S, (m_light, s_light), (m_dark, s_dark) = fld.field_builder(acq_num=acq_num)

    sn = abs(m_light - m_dark) / np.sqrt( s_light**2 + s_dark**2 )
    print(sn.min(), sn.max(), sn.mean())    

    xs, ys = S.pos
    sn = abs(m_light[xs, ys] - m_dark[xs, ys]) / np.sqrt( s_light[xs,ys]**2 + s_dark[xs,ys]**2 )
    print(sn.min(), sn.max(), sn.mean())    

    sci_frame = m_light - m_dark
    px = np.where(sci_frame < 0)
    fig, ax = plt.subplots(1,1)
    dpl.field_image(fig,ax,sci_frame)
    ax.set_title('Scientific Frame')
    plt.show()