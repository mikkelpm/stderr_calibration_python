import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from scipy.signal import lfilter
from pandas import read_csv
import matplotlib.pyplot as plt
from sequence_jacobian import hank
import sequence_jacobian.jacobian as jac
from stderr_calibration import MinDist

import sys


"""Estimation of HANK model
using functions produced by Auclert, Bardozcy, Rognlie & Straub (ECMA, 2021)
"""



"""Part 1a: Empirical moments (impulse responses)
"""

horzs = [0,1,2,8] # Response horizons (quarters after shock)

# TFP shock (Chang, Chen & Schorfheide, 2021, Figures 7 and 9)
data_ccs = read_csv('data/chang_chen_schorfheide.csv')
irf_z_data = {}
for v in ['TFP','GDP','BelowCutoff']:
    irf_z_data[v] = data_ccs[v+'_IRF_q50'].values[1:][horzs]/3
    irf_z_data[v+'_se'] = (data_ccs[v+'_IRF_q90'].values[1:][horzs]-data_ccs[v+'_IRF_q10'].values[1:][horzs])/3/(2*norm.ppf(0.9))
# Note: norm.ppf(0.9) is due to the paper reporting 80% credible bands
# We divide by 3 so we get 1 s.d. TFP shock
irf_z_data['BelowCutoff'] *= 100 # Change units to percentage points
irf_z_data['BelowCutoff_se'] *= 100

# MP shock (Miranda-Agrippino & Ricco, 2021, Figure 3)
data_mar = read_csv('data/mirandaagrippino_ricco.csv')
horzs_mth = np.array(horzs)*3 # Data is monthly, so we use last response in each quarter of interest
irf_mp_data = {}
for v in ['INDPRO','CPIAUCSL','GS1']:
    irf_mp_data[v] = data_mar[v+'_IRF_q50'].values[horzs_mth]
    irf_mp_data[v+'_se'] = (data_mar[v+'_IRF_q95'].values[horzs_mth]-data_mar[v+'_IRF_q05'].values[horzs_mth])/(2*norm.ppf(0.95))
# Note: norm.ppf(0.95) is due to the paper reporting 90% credible bands


"""Part 1b: Steady-state parameters
"""

# Steady-state calibration using same parameters as in ABRS (2021)
print('Computing steady state...')
ss = hank.hank_ss(beta_guess=.982, vphi_guess=0.786)
print('Done.')


""" Part 2: Model-based IRF functions
"""

T = 300 # Time horizon for equilibrium calculations

# Compute Jacobian of household block once and for all (only depends on non-estimated parameters)
print('Computing Jacobian of household block...')
J_ha = hank.household_trans.jac(ss, T, ['r', 'w', 'Div', 'Tax'])
ss.update({'J_ha': J_ha})
print('Done.')


# Compute general equilibrium Jacobians at specified parameter estimates
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


def get_irf(phi, kappa, ar1_z, ar2_z, sigma_z, ar1_mp, ar2_mp, horzs, ss, T):

    """Get impulse response functions of interest
    """

    # Compute Jacobians
    G = get_jacob(phi, kappa, ss, T)

    # TFP shock responses
    dgrowthZ = lfilter([1],[1,-ar1_z,-ar2_z],np.insert(np.zeros(T-1),0,sigma_z)) # Response of growth in TFP (following AR(2)) to shock, at each horizon
    dlogZ    = np.cumsum(dgrowthZ) # Response of log TFP at each horizon; for each horizon, the sum of current and past growth responses
    dZ       = ss['Z'] * dlogZ     # Response of level of TFP at each horizon, which is response of log level, dlogZ, times initial value (steady state)
    dlogY_dZ = dZ @ G['Y']['Z'][horzs,:].T / ss['Y'] # IRF of log output; divide response of level by starting value (steady state)
    dPctBelowGDP_dZ = dZ @ G['EARN_LT_GDP']['Z'][horzs,:].T # IRF of fraction of people earning less than GDP

    irf_z = {'Z': 100 * dlogZ[horzs], # IRF of log TFP
             'Y': 100 * dlogY_dZ,
             'PctBelowGDP': 100 * dPctBelowGDP_dZ}

    # MP shock responses
    dmp = lfilter([1],[1,-ar1_mp,-ar2_mp],np.insert(np.zeros(T-1),0,1)) # AR(2) shock to Taylor rule
    dlogY_dmp = dmp @ G['Y']['rstar'][horzs,:].T / ss['Y'] # IRF of log output
    dpi_dmp = dmp @ G['pi']['rstar'][horzs,:].T # IRF of inflation
    dlogP_dmp = np.cumsum(dpi_dmp) # IRF of log price level
    d1y_dmp = 4*(phi*dpi_dmp + dmp[horzs]) # IRF of 1-year nominal interest rate

    # Normalize MP shock responses
    irf_mp = {'Y': dlogY_dmp/d1y_dmp[0],
              'P': dlogP_dmp/d1y_dmp[0],
              'R1Y': d1y_dmp/d1y_dmp[0]}

    return irf_z, irf_mp


"""Part 3: Minimum distance estimation
"""

# Stack all moments
moment = np.hstack((irf_z_data['TFP'],irf_z_data['GDP'],irf_z_data['BelowCutoff'],
                    irf_mp_data['INDPRO'],irf_mp_data['CPIAUCSL'],irf_mp_data['GS1'][1:]))
moment_se = np.hstack((irf_z_data['TFP_se'],irf_z_data['GDP_se'],irf_z_data['BelowCutoff_se'],
                       irf_mp_data['INDPRO_se'],irf_mp_data['CPIAUCSL_se'],irf_mp_data['GS1_se'][1:]))
moment_num = len(moment)


def moment_fct(theta):

    """Model-implied moment function
    """

    (phi, kappa, ar1_z, ar2_z, sigma_z, ar1_mp, ar2_mp) = tuple(theta)
    irf_z, irf_mp = get_irf(phi, kappa, ar1_z, ar2_z, sigma_z, ar1_mp, ar2_mp, horzs, ss, T)
    return np.hstack((irf_z['Z'],irf_z['Y'],irf_z['PctBelowGDP'],irf_mp['Y'],irf_mp['P'],irf_mp['R1Y'][1:]))


# Estimation using diagonal weight matrix
param_bounds = [(0,np.inf),(1e-6,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)] # Parameter bounds
param_init = np.array([1.5,0.01,0,0,irf_z_data['TFP'][0]/100,0.5,0]) # Initial parameter guess

print('Running numerical optimization for diagonal weight matrix...')
opt_res = opt.minimize(lambda theta: np.sum(((moment-moment_fct(theta))/moment_se)**2),
                       param_init,
                       method='L-BFGS-B',
                       bounds=param_bounds,
                       callback=lambda x: print(np.array_str(x, precision=3, suppress_small=True)))
param_estim = opt_res['x'] # Parameter estimates
print('Done.')
assert opt_res['success']
print('Convergence message:')
print(opt_res['message'])


# Replication of optimisation
param_initbound = [(1,3),(-0.5,0.5),(-1,1),(-1,1),(0.5*irf_z_data['TFP'][0]/100,1.5*irf_z_data['TFP'][0]/100),(-1,1),(-1,1)]
param_lowerbound = [1,-0.5,-1,-1,0.5*irf_z_data['TFP'][0]/100,-1,-1]
param_dist = [2,1,2,2,irf_z_data['TFP'][0]/100,2,2]
param_initgrid = np.random.rand(10,7)
param_initgrid = np.multiply(param_initgrid,param_dist)+param_lowerbound
print('Replicating numerical optimization for diagonal weight matrix and using random initial guess...')
param_estim_repli = np.zeros(np.shape(param_initgrid))
success=np.zeros(np.shape(param_initgrid)[0])
for i,guess in enumerate(param_initgrid): 
    opt_res_repli = opt.minimize(lambda theta: np.sum(((moment-moment_fct(theta))/moment_se)**2),
                       guess,
                       method='L-BFGS-B',
                       bounds=param_bounds,
                       callback=None)
    param_estim_repli[i] = opt_res_repli['x'] # Parameter estimates
    success[i] = opt_res_repli['success']
print('Done.')


#Standard errors
obj = MinDist(moment_fct, moment, moment_se=moment_se)
print('Computing standard errors...')
res = obj.fit(param_estim=param_estim, weight_mat=np.diag(1/moment_se**2), eff=False)
print('Done.')


"""Part 4: Over-ID test
"""

res_overid = obj.overid(res)


"""Part 5: Efficient estimation
"""

res_eff = obj.fit(param_estim=param_estim, moment_jacob=res['moment_jacob'], eff=True)
nonzero_loadings = abs(res_eff['moment_loadings'])>=1e-4; # (Effectively) non-zero moment loadings


"""Part 6: Print output
"""

print('Parameter estimates (diagonal W):')
print(np.array_str(res['estim'], precision=3, suppress_small=True))
print('Worst-case standard errors (diagonal W):')
print(np.array_str(res['estim_se'], precision=3, suppress_small=True))
print('t-stats (diagonal W):')
print(np.array_str(res['estim']/res['estim_se'], precision=3, suppress_small=True))

print('Parameter estimates (efficient):')
print(np.array_str(res_eff['estim'], precision=3, suppress_small=True))
print('Worst-case standard errors (efficient):')
print(np.array_str(res_eff['estim_se'], precision=3, suppress_small=True))
print('t-stats (efficient):')
print(np.array_str(res_eff['estim']/res_eff['estim_se'], precision=3, suppress_small=True))

print('% reduction in standard errors caused by efficient weighting')
print(np.array_str(1-res_eff['estim_se']/res['estim_se'], precision=3, suppress_small=True))

print('Over-ID t-stats:')
print(res_overid['tstat'])
print('Joint p-value:')
print(res_overid['joint_pval'])
print('Joint test statistic:')
print(res_overid['joint_stat'])
print('Joint critical value (10%):')
print(res_overid['max_trace']*norm.ppf(0.95)**2)

print('Non-zero moment loadings (efficient):')
print(nonzero_loadings.astype(int))


"""Part 7: Plot IRFs
"""

# Compute all IRFs
h_max = 12
irf_z, irf_mp = get_irf(*param_estim, np.arange(h_max+1), ss, T)

# Plotting function
def my_plot(fig, plot_num, irf, horzs, data, moment_error, title):
    signif = 0.1
    ax = fig.add_subplot(2,3,plot_num)
    all_horzs = np.arange(len(irf))
    ax.plot(all_horzs, irf, label='Model')
    ax.errorbar(horzs, data, yerr=norm.ppf(1-signif/2)*moment_error,
                fmt='o', capsize=2.0, label='Data')
    ax.set_title(title, fontsize=10, fontweight='bold')
    plt.xticks(np.arange(0,len(irf)+1,4))

# Draw figures
nh = len(horzs)
plt.rcParams['font.size'] = '8'
fig = plt.figure()

my_plot(fig, 1, irf_z['Z'], horzs, irf_z_data['TFP'], res_overid['moment_error_se'][:nh],
        'TFP')
plt.ylabel('TFP shock', fontsize=10, fontweight='bold')

my_plot(fig, 2, irf_z['Y'], horzs, irf_z_data['GDP'], res_overid['moment_error_se'][nh:2*nh],
        'Output')

my_plot(fig, 3, irf_z['PctBelowGDP'], horzs, irf_z_data['BelowCutoff'], res_overid['moment_error_se'][2*nh:3*nh],
        '% Earn < GDP')

my_plot(fig, 4, irf_mp['Y'], horzs, irf_mp_data['INDPRO'], res_overid['moment_error_se'][3*nh:4*nh],
        'Output')
plt.ylabel('MP shock', fontsize=10, fontweight='bold')

my_plot(fig, 5, irf_mp['P'], horzs, irf_mp_data['CPIAUCSL'], res_overid['moment_error_se'][4*nh:5*nh],
        'Price Level')

my_plot(fig, 6, irf_mp['R1Y'], horzs, irf_mp_data['GS1'], np.insert(res_overid['moment_error_se'][5*nh:],0,0),
        '1-Year Bond Rate')
plt.legend()

fig.tight_layout()
plt.draw()

# Save figure
plt.savefig('irf.png', dpi=200)
plt.savefig('irf.eps', format='eps')
