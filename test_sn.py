import numpy as np
import matplotlib.pyplot as plt
import skysimulation.display as dpl
import skysimulation.field as fld
import skysimulation.restoration as rst





if __name__ == '__main__':

    acq_num = 5
    S, (m_light, s_light), (m_dark, s_dark) = fld.field_builder(acq_num=acq_num)
    
    sn = abs(m_light - m_dark) / np.sqrt( s_light**2 + s_dark**2 )
    print(sn.min(), sn.max(), sn.mean())    

