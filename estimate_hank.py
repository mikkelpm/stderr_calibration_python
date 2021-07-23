import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from scipy.signal import lfilter
from sequence_jacobian import hank
import sequence_jacobian.jacobian as jac
from stderr_calibration import MinDist


"""Estimation of HANK model
using functions produced by Auclert, Bardozcy, Rognlie & Straub (ECMA, 2021)
"""


"""Part 1a: Steady-state parameters
"""

# Steady-state calibration using same parameters as in ABRS (2021)
print('Computing steady state...')
ss = hank.hank_ss(beta_guess=.982, vphi_guess=0.786)
print('Done.')
T = 300 # Time horizon for equilibrium calculations


"""Part 1b: Empirical moments (impulse responses)
"""

horzs = [0,4,8] # Response horizons
# TFP shock (Chang, Chen & Schorfheide, 2021)
irf_z_data =  {'Z': np.array([2,2.1,2.05]), # TFP
               'Z_se': np.array([0.2,0.7,0.7])/norm.ppf(0.9)/2,
               'Y': np.array([0.8,1.8,1.7]), # Output
               'Y_se': np.array([0.2,0.8,0.8])/norm.ppf(0.9)/2,
               'P10': np.array([0.23,0.235,0.228])-0.225, # 10th percentile of earnings
               'P10_se': np.array([0.015,0.01,0.005])/norm.ppf(0.9)/2,
               'P90': np.array([2.49,2.5,2.5])-2.5, # 90th percentile of earnings
               'P90_se': np.array([0.03,0.015,0.005])/norm.ppf(0.9)/2}
# Note: norm.ppf(0.9) is due to the paper reporting 80% credible bands

# MP shock (Miranda-Agrippino & Ricco, 2021)
irf_mp_data = {'Y': np.array([-1,-1.6,-1.4]), # Output
               'Y_se': np.array([0.5,1.3,1.8])/norm.ppf(0.95)/2,
               'P': np.array([-0.25,-0.5,-0.65]), # Price level
               'P_se': np.array([0.17,0.5,0.7])/norm.ppf(0.95)/2,
               '1Y': np.array([1,0.15,0]), # 1-year bond rate
               '1Y_se': np.array([0,0.45,0.4])/norm.ppf(0.95)/2}


""" Part 2: model-based IRF functions
"""

# Compute Jacobian of household block once and for all (only depends on non-estimated parameters)
print('Computing Jacobian of household block...')
J_ha = jac.get_G([hank.household_trans], ['r', 'w'], [], [], T, ss, save=True)
ss.update({'J_ha': J_ha})
print('Done.')


def get_jacob(phi, kappa, ss, T):
    
    """Get het agent Jacobian
    """
    
    exogenous = ['rstar', 'Z']
    unknowns = ['pi', 'w', 'Y']
    targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']
    block_list = [ss['J_ha'], hank.firm, hank.monetary, hank.fiscal, hank.nkpc, hank.mkt_clearing] 
    ss2 = ss.copy()
    ss2.update({'phi': phi, 'kappa': kappa}) # Update non-steady-state parameters
    G = jac.get_G(block_list, exogenous, unknowns, targets, T, ss2)
    return G
    

def get_irf(phi, kappa, ar_z, ma_z, sigma_z, ar_mp, ma_mp, horzs, ss, T):
    
    """Get impulse response functions of interest
    """
    
    # Compute Jacobians
    G = get_jacob(phi, kappa, ss, T)
    # It would be faster to reuse parts of the Jacobians, see ARBS (2021, Section 5.4)
    
    # TFP shock responses
    dz = ss['Z'] * lfilter([1,ma_z],[1,-ar_z],np.insert(np.zeros(T-1),0,sigma_z)) # ARMA(1,1) shock to TFP
    irf_z = {'Z': dz[horzs] / ss['Z'], # IRF of TFP
             'Y': (dz @ G['Y']['Z'][horzs,:].T) / ss['Y']} # IRF of output
    
    # MP shock responses
    dmp = lfilter([1,ma_mp],[1,-ar_mp],np.insert(np.zeros(T-1),0,1)) # ARMA(1,1) shock to Taylor rule
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

# Macro moments
moment_macro = np.hstack((irf_z_data['Z'],irf_z_data['Y'],
                          irf_mp_data['Y'],irf_mp_data['P'],irf_mp_data['1Y'][1:]))
moment_macro_se = np.hstack((irf_z_data['Z_se'],irf_z_data['Y_se'],
                             irf_mp_data['Y_se'],irf_mp_data['P_se'],irf_mp_data['1Y_se'][1:]))
moment_macro_num = len(moment_macro)

# Micro moments
moment_micro_num = 0

# Stack all moments
moment = moment_macro
moment_se = moment_macro_se
moment_num = len(moment)


def moment_fct(theta, horzs, ss, T):
    
    """Model-implied moment function
    """
    
    (phi, kappa, ar_z, ma_z, sigma_z, ar_mp, ma_mp) = tuple(theta)
    irf_z, irf_mp = get_irf(phi, kappa, ar_z, ma_z, sigma_z, ar_mp, ma_mp, horzs, ss, T)
    return np.hstack((irf_z['Z'],irf_z['Y'],irf_mp['Y'],irf_mp['P'],irf_mp['1Y'][1:]))


def mindist_obj(theta, moment, horzs, ss, T):
    
    """Minimum distance objective function
    """
    
    quadf = lambda a: a @ weight_mat_macro @ a
    return quadf(moment-moment_fct(theta,horzs,ss,T))
    

# Estimation using only macro moments
weight_mat_macro = np.diag(np.hstack((1/moment_macro_se**2,np.zeros(moment_micro_num)))) # Weight matrix (zeros on micro moments)
param_bounds = [(1,np.inf),(-np.inf,np.inf),(-1,1),(-np.inf,np.inf),(0,np.inf),(-1,1),(-np.inf,np.inf)] # Parameter bounds
param_init = np.array([1.5,0.01,0.95,0,irf_z_data['Z'][0],0.7,0]) # Initial parameter guess

print('Running numerical optimization for macro-only estimation...')
opt_res = opt.minimize(lambda theta: mindist_obj(theta, moment, horzs, ss, T),
                       param_init,
                       method='L-BFGS-B',
                       bounds=param_bounds,
                       callback=lambda x: print(np.round(x,3)))
param_estim = opt_res['x'] # Parameter estimates
print('Done.')


"""Part 4: Standard errors
"""

obj = MinDist(lambda theta: moment_fct(theta, horzs, ss, T), moment, moment_se=moment_se)
print('Computing standard errors...')
res = obj.fit(param_estim=param_estim, weight_mat=weight_mat_macro, eff=False)
print('Done.')

print('Parameter estimates:')
print(res['estim'])
print('Worst-case standard errors:')
print(res['estim_se'])
#res_eff = obj.fit(param_estim=theta_estim, weight_mat=weight_mat, moment_jacob=res['moment_jacob'], eff=True)
