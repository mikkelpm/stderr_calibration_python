import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from sequence_jacobian import hank
import sequence_jacobian.jacobian as jac
from stderr_calibration import MinDist


"""Estimation of HANK model
using functions provided by Auclert, Bardozcy, Rognlie & Straub (ECMA, 2021)
"""


"""Part 1: Settings and empirical moments
"""

# Steady-state calibration as in ABRS (2021) paper
ss = hank.hank_ss(beta_guess=.982, vphi_guess=0.786)
T = 300

# Empirical moments
horzs = [0,4,8] # Response horizons
irf_z_data =  {'Z': np.array([2,2.1,2.05]),
               'Z_se': np.array([0.2,0.7,0.7])/norm.ppf(0.9)/2,
               'Y': np.array([0.8,1.8,1.7]),
               'Y_se': np.array([0.2,0.8,0.8])/norm.ppf(0.9)/2} # TFP shock (Chen, Chang & Schorfheide, 2021)
irf_mp_data = {'Y': np.array([-1,-1.6,-1.4]),
               'Y_se': np.array([0.5,1.3,1.8])/norm.ppf(0.95)/2,
               'P': np.array([-0.25,-0.5,-0.65]),
               'P_se': np.array([0.17,0.5,0.7])/norm.ppf(0.95)/2,
               '1Y': np.array([1,0.15,0]),
               '1Y_se': np.array([0,0.45,0.4])/norm.ppf(0.95)/2} # MP shock (Miranda-Agrippino & Ricco, 2021)


""" Part 2: model-based IRF functions
""" 

def get_jacob(phi, kappa, ss, T):
    
    """Get het agent Jacobian
    """
    
    exogenous = ['rstar', 'Z']
    unknowns = ['pi', 'w', 'Y']
    targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']
    block_list = [hank.firm, hank.monetary, hank.fiscal, hank.nkpc, hank.mkt_clearing, hank.household] 
    ss.update({'phi': phi, 'kappa': kappa}) # Update non-steady-state parameters
    G = jac.get_G(block_list, exogenous, unknowns, targets, T, ss, save=False)
    return G
    

def get_irf(phi, kappa, rho_z, sigma_z, rho_mp, horzs, ss, T):
    
    """Get impulse response functions of interest
    """
    
    # Compute Jacobians
    G = get_jacob(phi, kappa, ss, T)
    
    # TFP shock
    dz = ss['Z'] * sigma_z * rho_z ** np.arange(T) # Shock to TFP
    irf_z = {'Z': dz[horzs] / ss['Z'], # IRF of TFP
             'Y': (dz @ G['Y']['Z'][horzs,:].T) / ss['Y']} # IRF of output
    
    # MP shock
    dmp = rho_mp ** np.arange(T) # Shock to Taylor rule
    dY_dmp = dmp @ G['Y']['rstar'][horzs,:].T / ss['Y'] # IRF of output
    dpi_dmp = dmp @ G['pi']['rstar'][horzs,:].T # IRF of inflation
    dP_dmp = np.cumsum(dpi_dmp) # IRF of price level
    d1y_dmp = 4*(phi*dpi_dmp + dmp[horzs]) # IRF of 1-year nominal interest rate
    
    # Normalize MP shock responses
    irf_mp = {'Y': dY_dmp/d1y_dmp[0],
              'P': dP_dmp/d1y_dmp[0],
              '1Y': d1y_dmp/d1y_dmp[0]}
    
    return irf_z, irf_mp


"""Part 3: Minimum distance estimation
"""

# Stack empirical moments and their standard errors
moments = np.hstack((irf_z_data['Y'],
                     irf_mp_data['Y'],irf_mp_data['P'],irf_mp_data['1Y'][1:]))
moments_se = np.hstack((irf_z_data['Y_se'],
                        irf_mp_data['Y_se'],irf_mp_data['P_se'],irf_mp_data['1Y_se'][1:]))

# Model-implied moment function
def h(theta, horzs, ss, T):
    (phi, kappa, rho_z, sigma_z, rho_mp) = tuple(theta)
    irf_z, irf_mp = get_irf(phi, kappa, rho_z, sigma_z, rho_mp, horzs, ss, T)
    return np.hstack((irf_z['Y'],irf_mp['Y'],irf_mp['P'],irf_mp['1Y'][1:]))

# Run estimation
weight_mat = np.diag(1/moments_se**2)
quadf = lambda a: a @ weight_mat @ a
mindist_obj = lambda theta: quadf(moments-h(theta,horzs,ss,T))
param_bounds = [(1,np.inf),(-np.inf,np.inf),(-1,1),(0,np.inf),(-1,1)]
theta_init = np.array([1.5,0.01,(irf_z_data['Y'][2]/irf_z_data['Y'][1])**(1/(horzs[2]-horzs[1])),irf_z_data['Y'][0],(irf_mp_data['1Y'][1]/irf_mp_data['1Y'][0])**(1/horzs[1])])
opt_res = opt.minimize(mindist_obj, theta_init, method='SLSQP', bounds=param_bounds, options={'disp': True})
theta_estim = opt_res['x']


"""Part 4: Standard errors
"""

obj = MinDist(lambda theta: h(theta, horzs, ss, T), moments, moment_se=moments_se)
res = obj.fit(param_estim=theta_estim, weight_mat=weight_mat, eff=False)
res_eff = obj.fit(param_estim=theta_estim, weight_mat=weight_mat, moment_jacob=res['moment_jacob'], eff=True)
