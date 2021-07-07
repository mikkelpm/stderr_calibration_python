import numpy as np
from scipy.linalg import block_diag

import sys
sys.path.insert(1,'../')
from stderr_calibration import MinDist


"""Unit tests of minimum distance inference functions
Intended for use with the "pytest" testing framework
"""


# Define small-scale problem used for some tests

G = np.array([[1,0],[1,1],[0,2]])
h = lambda x: x @ G.T
theta = np.array([1,1])
mu = h(theta)
sigma = np.array([1,2,0.5])

V_fullinfo = sigma.reshape(-1,1) * np.array([[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]]) * sigma

V_blockdiag = V_fullinfo.copy()
V_blockdiag[0,1:] = np.nan
V_blockdiag[1:,0] = np.nan


# Define test cases

def test_closedform():
    
    """Test closed-form formulas
    """
    
    obj = MinDist(h,mu,moment_se=sigma)
    
    # Estimation with default weight matrix
    res = obj.fit(opt_init=np.array(np.zeros(theta.shape)), eff=False)
    W = np.diag(1/sigma**2)
    aux = np.linalg.solve(G.T @ W @ G, G.T @ W).T
    np.testing.assert_allclose(res['moment_loadings'], aux)
    np.testing.assert_allclose(res['transf_estim_se'], sigma @ np.abs(aux))
    
    # Efficient estimation (see formulas in paper appendix)
    res_eff = obj.fit(opt_init=np.array(np.zeros(theta.shape)))
    if sigma[0]*np.abs(G[1,0]*G[2,1]) <= sigma[1]*np.abs(G[0,0]*G[2,1])+sigma[2]*np.abs(G[0,0]*G[1,1]):
        x = np.array([1/G[0,0], 0, 0])
    else:
        x = np.array([0, 1/G[1,0], -G[1,1]/(G[1,0]*G[2,1])])
    np.testing.assert_allclose(x @ mu, res_eff['transf_estim'][0], rtol=1e-6)
    np.testing.assert_allclose(abs(x) @ sigma, res_eff['transf_estim_se'][0], rtol=1e-6)


def test_diag():
    
    """Test basic functionality with known diagonal
    """
    
    def run_tests(obj):
        
        # Estimation output
        res = obj.fit(opt_init=np.array(np.zeros(theta.shape)), eff=False)
        assert res['moment_loadings'].shape==G.shape
        np.testing.assert_allclose(res['moment_fit'], mu)
        np.testing.assert_allclose(res['moment_jacob'], G)
        np.testing.assert_allclose(res['transf_estim'], theta)
        np.testing.assert_array_less(-res['transf_estim_se'], 0)
        
        # Test output
        test_res = obj.test(res)
        np.testing.assert_array_less(-test_res['tstat'], 0)
        assert test_res['joint_stat']>0
        overid_res = obj.overid(res)
        np.testing.assert_allclose(overid_res['moment_error'], 0, atol=1e-7)
        np.testing.assert_array_less(-overid_res['tstat'], 0)
        assert overid_res['joint_stat']>0
        
        # Efficient estimation
        res_eff = obj.fit(opt_init=np.zeros(theta.shape))
        np.testing.assert_allclose(res['transf_estim'], theta)
        
        return res, res_eff
    
    # Limited information
    obj = MinDist(h,mu,moment_se=sigma)
    res, res_eff = run_tests(obj)
    for i in range(len(theta)):
        np.testing.assert_allclose(np.diag(res['worstcase_varcov'][i]), sigma**2)
    np.testing.assert_array_less(res_eff['transf_estim_se'], res['transf_estim_se'])
    
    # Test that full optimization gives the same as one-step (due to linear moment function)
    res2_eff = obj.fit(opt_init=np.zeros(theta.shape),one_step=False)
    np.testing.assert_allclose(res_eff['transf_estim'], res2_eff['transf_estim'], rtol=1e-5)
    np.testing.assert_allclose(res_eff['transf_estim_se'], res2_eff['transf_estim_se'], rtol=1e-5)
    
    # Full information
    obj_fullinfo = MinDist(h,mu,moment_varcov=V_fullinfo)
    res_fullinfo, res_eff_fullinfo = run_tests(obj_fullinfo)
    
    np.testing.assert_array_less(res_fullinfo['transf_estim_se'], res['transf_estim_se'])
    np.testing.assert_array_less(res_eff_fullinfo['transf_estim_se'], res_eff['transf_estim_se'])

    
def test_psd():
    
    """Test that positive semidefinite problem gives right solution
    in simple case with known diagonal"""
    
    obj = MinDist(h,mu,moment_se=sigma)
    res = obj.fit(opt_init=np.array(np.zeros(theta.shape)), eff=False)
    obj.diag_only=False # Force PSD programming
    res2 = obj.fit(opt_init=np.array(np.zeros(theta.shape)), eff=False)
    np.testing.assert_allclose(res2['transf_estim_se'],res['transf_estim_se'], rtol=1e-5)
    for i in range(len(theta)):
        np.testing.assert_allclose(res2['worstcase_varcov'][i], res['worstcase_varcov'][i], rtol=1e-5)
    

def test_blockdiag():
    
    """Test known block diagonal
    """
    
    obj = MinDist(h,mu,moment_varcov=V_blockdiag)
    res = obj.fit(opt_init=np.array(np.zeros(theta.shape)), eff=False)
    res_eff = obj.fit(opt_init=np.array(np.zeros(theta.shape)))
    obj.blockdiag_only=False # Force PSD programming
    res2 = obj.fit(opt_init=np.array(np.zeros(theta.shape)), eff=False)
    res2_eff = obj.fit(opt_init=np.array(np.zeros(theta.shape)))
    
    np.testing.assert_array_less(res_eff['transf_estim_se'], res['transf_estim_se'])
    np.testing.assert_allclose(res['transf_estim_se'], res2['transf_estim_se'], rtol=1e-5)
    np.testing.assert_allclose(res_eff['transf_estim_se'], res2_eff['transf_estim_se'], rtol=3e-2)
    

def test_highdim():
    
    
    """Test high-dimensional example
    """

    # Generate random high-dimensional problem with known block diagonal
    blocks_num = 4 # Number of blocks; 1st block is 1x1, 2nd block is 2x2, etc.
    k = 4 # Number of parameters
    np.random.seed(123) # Random seed
    G = np.random.randn(int(blocks_num*(blocks_num+1)/2),k)
    h = lambda x: x @ G.T
    mu = np.arange(k) @ G.T
    x = [np.random.randn(i+1,i+1) for i in range(blocks_num)]
    V = block_diag(*[a @ a.T for a in x])
    V[V==0] = np.nan
    
    # Estimate
    obj = MinDist(h,mu,moment_varcov=V)
    res = obj.fit(opt_init=np.zeros(k),eff=False)
    res_eff = obj.fit(opt_init=np.zeros(k))
    obj.blockdiag_only=False # Force PSD programming
    res2 = obj.fit(opt_init=np.zeros(k),eff=False)
    res2_eff = obj.fit(opt_init=np.zeros(k))
    
    np.testing.assert_array_less(res_eff['transf_estim_se'], res['transf_estim_se'])
    np.testing.assert_allclose(res['transf_estim_se'], res2['transf_estim_se'], rtol=1e-5)
    np.testing.assert_allclose(res_eff['transf_estim_se'], res2_eff['transf_estim_se'], rtol=5e-2)

