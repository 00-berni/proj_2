from .display import *
from .stuff import *
from .field import initialize, field_builder, MIN_m, MAX_m, M_SEED, POS_SEED, K, N, M, ALPHA, BETA, Star
from .field import from_parms_to_distr, BACK_SEED,BACK_MEAN,BACK_SIGMA
from .field import atm_seeing, noise, SEEING_SIGMA, NOISE_SEED, NOISE_MEAN, NOISE_SIGMA
from .restoration import bkg_est, LR_deconvolution,searching,kernel_estimation,light_recover

__author__  = '00-berni'
__version__ = '0.0.0'

MASS_ENDS = [MIN_m, MAX_m]
FRAME = {'size': N, 'stars': M, 'mass_seed': M_SEED, 'mass_range': MASS_ENDS, 'pos_seed': POS_SEED, 'norm': K}
BACK  = {'seed': BACK_SEED, 'mean': BACK_MEAN, 'sigma': BACK_SIGMA}
NOISE = {'seed': NOISE_SEED, 'mean': NOISE_MEAN, 'sigma': NOISE_SIGMA}
