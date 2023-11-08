import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from .display import fast_image
from .field import N, noise, Uniform, Gaussian, check_field

##*
def dark_elaboration(distr: Uniform | Gaussian, iteration: int = 3, dim: int = N, display_fig: bool = False) -> np.ndarray:
    """The function computes a number (`iteration`) of darks
    and averages them in order to get a mean estimation 
    of the detector noise

    :param n_value: detector noise, defaults to 3e-4
    :type n_value: float, optional
    :param iteration: number of darks to compute, defaults to 3
    :type iteration: int, optional

    :return: mean dark
    :rtype: np.ndarray
    """
    # generating the first dark
    dark = noise(distr, dim=dim)
    # making the loop
    for i in range(iteration-1):
        dark += noise(distr, dim=dim)
    # averaging
    dark /= iteration
    if display_fig:
        fast_image(dark,v=1,title=f'Dark elaboration\nAveraged on {iteration} iterations')
    return dark

def bkg_est(field: np.ndarray, display_fig: bool = False) -> float:
    field = np.copy(field).flatten()
    field = field[np.where(field > 0)[0]]
    num = np.sqrt(len(field))
    bins = np.arange(np.log10(field).min(),np.log10(field).max(),1/num)
    counts, bins = np.histogram(np.log10(field),bins=bins)
    tmp = counts[counts.argmax()+1:]
    dist = abs(tmp[:-2] - tmp[2:])
    pos = np.where(counts == tmp[dist.argmax()+2])[0]
    mbin = (max(bins[pos])+bins[counts.argmax()])/2
    if display_fig:
        plt.figure(figsize=(14,10))
        plt.stairs(counts, bins,fill=True)
        # plt.axvline(max(bins[pos]),0,1,linestyle='--',color='orange')
        # plt.axvline(bins[counts.argmax()],0,1,linestyle='--',color='red')
        plt.axvline(mbin,0,1,linestyle='--',color='orange')
        plt.xlabel('$\\log_{10}(F_{sn})$')
        plt.ylabel('counts')
        plt.show()
    return mbin


def detection(field: np.ndarray, back: float):
    flat = field.flatten()
    pks, _ = find_peaks(flat,back)
    print(len(pks))
    pos = np.where(flat > back)[0]
    print(pos)
    plt.figure()
    plt.plot(np.arange(len(flat)),flat)
    plt.plot(pks,flat[pks],'.')
    plt.show()