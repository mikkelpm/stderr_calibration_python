import numpy as np
import numdifftools as nd
import scipy.optimize as opt
import cvxopt as cvx
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import chi2, norm


"""Worst-case standard errors for minimum distance estimates
without knowledge of the correlation matrix for the matched moments
    
If desired, also computes:
- worst-case efficient estimates
- full-information efficient estimates
- over-identification test for each individual moment
    
Reference:
Cocci, Matthew D. & Mikkel Plagborg-Moller, "Standard Errors for Calibrated Parameters"
https://scholar.princeton.edu/mikkelpm/calibration
"""

class MinDist:
    
    
    def __init__(self, moment_fct, moment_estim,
                 moment_se=None, moment_varcov=None, moment_fct_deriv=None,
                 zero_thresh=1e-8):
        
        self.moment_fct = moment_fct
        self.moment_estim = np.asarray(moment_estim).flatten()
        self.moment_se = np.asarray(moment_se).flatten()
        self.moment_varcov = np.asarray(moment_varcov)
        self.zero_thresh = zero_thresh
        self.moment_num = len(moment_estim)
        self.full_info = (moment_varcov is not None)
        self.moment_fct_deriv = self._deriv(moment_fct_deriv, self.moment_fct)
        
        # Check inputs
        assert (moment_se is not None) or (moment_varcov is not None)
        assert (moment_se is None) or (len(self.moment_se)==self.moment_num)
        assert (moment_varcov is None) or (self.moment_varcov.shape==(self.moment_num,self.moment_num))
        assert np.all(np.isreal(self.moment_estim))
        assert np.all(np.isreal(self.moment_se))
        assert np.all(np.isreal(self.moment_varcov))
        assert np.isscalar(self.zero_thresh) and (self.zero_thresh>=0)
        assert self.moment_num>=1
        
    
    def fit(self, transf=lambda x: x, weight_mat=None,
            opt_init=None, estim_fct=None, eff=True, one_step=True, transf_deriv=None,
            param_estim=None, transf_estim=None, transf_jacob=None, moment_fit=None, moment_jacob=None):
        
        """Minimum distance estimates and standard errors,
        either with full-information moment var-cov matrix
        or with limited-information individual moment variances
        """
        
        # Check inputs
        assert (param_estim is not None) or (opt_init is not None) or (estim_fct is not None)
        
        # Transformation Jacobian function
        transf_deriv = self._deriv(transf_deriv, transf)
        
        # Determine weight matrix, if not supplied
        if self.full_info and (eff or (weight_mat is None)):
            weight_mat = np.linalg.inv(self.moment_varcov) # Full-info efficient weight matrix
        if weight_mat is None:
            weight_mat = np.diag(self.moment_se**(-2)) # Ad hoc diagonal weight matrix
        
        # Default estimation routine
        if estim_fct is None:
            estim_fct = lambda W: opt.fmin_bfgs(lambda x: (self.moment_estim-self.moment_fct(x)) @ W @ np.reshape(self.moment_estim-self.moment_fct(x),(-1,1)), opt_init)
        
        # Initial estimate of parameters, if not supplied
        if param_estim is None:
            param_estim = estim_fct(weight_mat)
        param_num = len(param_estim)
        
        # Transformation, moment function, and Jacobians at initial estimate
        transf_estim, transf_jacob, moment_fit, moment_jacob \
            = self.estim_update(param_estim, transf, transf_deriv,
                                transf_estim=transf_estim, transf_jacob=transf_jacob,
                                moment_fit=moment_fit, moment_jacob=moment_jacob)
        moment_loadings = self._get_moment_loadings(moment_jacob, weight_mat, transf_jacob)
        transf_num = 1 if np.isscalar(transf_estim) else len(transf_estim)
        
        # Efficient estimates
        if eff:
            
            if self.full_info: # Full information
            
                if one_step: # One-step estimation
                    param_estim = self._get_onestep(moment_fit, weight_mat, moment_jacob, param_estim)
                    transf_estim, transf_jacob, moment_fit, moment_jacob = self.estim_update(param_estim, transf, transf_deriv)
                else: # Full optimization
                    # Do nothing, since param_estim already contains estimates of interest
                    pass
                
            else: # Limited information
            
                if transf_num > 1: # If more than one parameter of interest, handle each separately by recursive call
                
                    transf_estim_init = transf_estim.copy()
                    transf_estim = np.empty(transf_num)
                    transf_estim_se = np.empty(transf_num)
                    moment_loadings = np.empty((self.moment_num,transf_num))
                    for i in range(transf_num):
                        the_res = self.fit(transf=lambda x: transf(x)[i],
                                           weight_mat=weight_mat,
                                           estim_fct=estim_fct,
                                           eff=True, one_step=one_step,
                                           param_estim=param_estim,
                                           transf_estim=transf_estim_init[i],
                                           transf_jacob=transf_jacob[i,:],
                                           moment_fit=moment_fit,
                                           moment_jacob=moment_jacob) # Computations for scalar parameter of interest
                        transf_estim[i] = the_res['transf_estim']
                        transf_estim_se[i] = the_res['transf_estim_se']
                        moment_loadings[:,i] = the_res['moment_loadings']
                        
                else: # If only single parameter of interest
                
                    transf_estim_se, moment_loadings = self.worstcase_eff(moment_jacob, transf_jacob)
                    if one_step: # One-step estimation
                        transf_estim = self._get_onestep(moment_fit, None, moment_loadings, transf_estim)
                    else: # Full optimization estimation
                        # Weight matrix puts weight on only k moments, where k=#parameters
                        sort_inds = np.abs(moment_loadings).argsort()
                        weight_mat_new = weight_mat.copy()
                        weight_mat_new[sort_inds[:self.moment_num-param_num],:] = 0
                        weight_mat_new[:,sort_inds[:self.moment_num-param_num]] = 0
                        param_estim = estim_fct(weight_mat_new)
                        transf_estim = transf(param_estim)
        
        # Start building results dictionary
        res = {'transf_estim': transf_estim,
               'param_estim': param_estim,
               'weight_mat': weight_mat if (self.full_info or (not eff)) else None,
               'moment_fit': moment_fit,
               'moment_jacob': moment_jacob,
               'moment_loadings': moment_loadings,
               'transf_jacob': transf_jacob,
               'transf_num': transf_num}
        
        # Standard errors
        if self.full_info: # Full information
            transf_estim_varcov = moment_loadings.T @ self.moment_varcov @ moment_loadings
            transf_estim_se = np.sqrt(np.diag(transf_estim_varcov))
            res['transf_estim_varcov'] = transf_estim_varcov
        else: # Limited information
            if eff:
                # Do nothing, since standard errors have already been computed above
                pass
            else:
                transf_estim_se = np.abs(moment_loadings.T) @ self.moment_se.reshape(-1,1)
        
        res['transf_estim_se'] = transf_estim_se
        
        return res
    
    
    def test(self, estim_res, joint=True, test_weight_mat=None):
        
        """Test whether transformed parameters equal zero
        """
        
        # t-statistics
        tstat = estim_res['transf_estim']/estim_res['transf_estim_se']
        tstat_pval = 2*norm.cdf(-np.abs(tstat))
        res = {'estim_res': estim_res,
               'tstat': tstat,
               'tstat_pval': tstat_pval}
        
        if not joint:
            return res
        
        # Weight matrix for joint test statistic
        if test_weight_mat is None:
            if self.full_info: # Full information
                test_weight_mat = np.linalg.inv(estim_res['transf_estim_varcov'])
            else: # Limited information
                # Ad hoc choice motivated by independence
                test_weight_mat = np.linalg.inv(estim_res['moment_loadings'].T @ np.diag(self.moment_se**2) @ estim_res['moment_loadings'])
        
        # Check dimensions
        assert test_weight_mat.shape == (estim_res['transf_num'],estim_res['transf_num'])
        
        # Test statistic
        joint_stat = estim_res['transf_estim'] @ test_weight_mat @ estim_res['transf_estim'].reshape(-1,1)
        
        # p-value
        if self.full_info:
            joint_pval = 1-chi2.cdf(joint_stat, estim_res['transf_num'])
        else:
            max_trace, max_trace_varcov = self.solve_sdp(estim_res['moment_loadings'] @ test_weight_mat @ estim_res['moment_loadings'].T)
            joint_pval = 1-chi2.cdf(joint_stat/max_trace, 1)
            if joint_pval>0.215: # Test can only be used at significance levels < 0.215
                joint_pval = np.array([1])
            res['max_trace'] = max_trace
            res['max_trace_varcov'] = np.array(max_trace_varcov)
        
        res.update({'test_weight_mat': test_weight_mat,
                    'joint_stat': joint_stat.item(),
                    'joint_pval': joint_pval.item()})
        
        return res
    
    
    def overid(self, estim_res, joint=True):
        
        """Over-identification test
        """
        
        # Errors in fitting moments
        moment_error = self.moment_estim - estim_res['moment_fit']
        
        # Standard errors for moment errors
        M = np.eye(self.moment_num) - self._get_moment_loadings(estim_res['moment_jacob'],
                                                                estim_res['weight_mat'],
                                                                estim_res['moment_jacob']).T
        the_estim_res = self.fit(lambda x: x,
                                 eff=False,
                                 weight_mat=np.eye(self.moment_num),
                                 param_estim=estim_res['param_estim'],
                                 transf_estim=moment_error,
                                 transf_jacob=M,
                                 moment_fit=self.moment_estim,
                                 moment_jacob=np.eye(self.moment_num))
        '''Only the inputs "weight_mat", "transf_jacob", and "moment_jacob" are
        actually used to calculate the standard errors - the other inputs are
        only provided to avoid unnecessary computations
        '''
        
        # Test statistic and p-value
        the_test_res = self.test(the_estim_res, joint=joint, test_weight_mat=estim_res['weight_mat'])
        
        res = {'moment_error': moment_error,
               'moment_error_se': the_estim_res['transf_estim_se'],
               'tstat': the_test_res['tstat'],
               'tstat_pval': the_test_res['tstat_pval']}
        
        if joint:
            res.update({'joint_stat': the_test_res['joint_stat'],
                        'joint_pval': the_test_res['joint_pval']})
            if self.full_info: # Adjust degrees of freedom
                res['joint_pval'] = 1-chi2.cdf(the_test_res['joint_stat'], self.moment_num-len(estim_res['param_estim']))
    
        return res
        
    
    def worstcase_eff(self, moment_jacob, transf_jacob):
        
        """Compute worst-case efficient moment loadings via median regression
        See main paper for explanation
        """
        
        # Set up median regression as described in paper
        GpG = moment_jacob.T @ moment_jacob
        Y = self.moment_se.reshape(-1,1) * (moment_jacob @ np.linalg.solve(GpG, transf_jacob.reshape(-1,1)))
        val, vec = np.linalg.eig(np.eye(self.moment_num) - moment_jacob @ np.linalg.solve(GpG, moment_jacob.T))
        moment_jacob_perp = vec[:, abs(val)>self.zero_thresh]
        X = -self.moment_se.reshape(-1,1) * moment_jacob_perp
    
        # Run median regression
        qr_mod = QuantReg(Y, X)
        qr_fit = qr_mod.fit(q=.5)
        resid = qr_fit._results.resid # Residuals
        
        # Efficient moment loadings and standard errors
        moment_loadings = resid / self.moment_se
        se = np.abs(resid).sum()
        
        return se, moment_loadings
    
    
    def solve_sdp(self, A):
        
        """Solve semidefinite programming problem
        max tr(A*V) s.t. V psd and diag(V)=self.moment_se**2
        """
        
        # Coefficient matrices
        c = cvx.matrix(-self.moment_se.reshape(-1,1)**2,tc='d')
        G = cvx.sparse(cvx.matrix(np.eye(self.moment_num**2)[:,np.eye(self.moment_num).flatten()==1]))
        h = cvx.matrix(-A,tc='d')
        
        # Solve SDP
        sol = cvx.solvers.sdp(c,Gs=[G],hs=[h])
        
        # Return objective value and optimal V
        return sol['dual objective'], sol['zs'][0]
    
    
    def estim_update(self, param_estim, transf, transf_deriv,
                     transf_estim=None, transf_jacob=None, moment_fit=None, moment_jacob=None):
            
            """Update estimated parameter transformation and moments,
            including their Jacobians.
            Avoids recomputing quantities if they're already supplied.
            """
            
            if transf_estim is None:
                transf_estim = transf(param_estim)
            if transf_jacob is None:
                transf_jacob = transf_deriv(param_estim)
            if moment_fit is None:
                moment_fit = self.moment_fct(param_estim)
            if moment_jacob is None:
                moment_jacob = self.moment_fct_deriv(param_estim)
            return transf_estim, transf_jacob, moment_fit, moment_jacob
    
    
    def _get_onestep(self, moment_init, weight_mat, moment_jacob, param_init):
        
        """One-step estimation
        """
        
        if weight_mat is None:
            subtr = moment_jacob.T @ np.reshape(moment_init-self.moment_estim,(-1,1))
        else:
            subtr = np.linalg.solve(moment_jacob.T @ weight_mat @ moment_jacob, moment_jacob.T @ weight_mat @ np.reshape(moment_init-self.moment_estim,(-1,1)))
        return param_init - subtr.flatten()
    
    
    @staticmethod
    def _get_moment_loadings(moment_jacob, weight_mat, transf_jacob):
        
        """Asymptotic loadings of minimum distance estimator on empirical moments
        """
        
        return weight_mat @ moment_jacob @ np.linalg.solve(moment_jacob.T @ weight_mat @ moment_jacob, transf_jacob.T)
    
    
    @staticmethod
    def _deriv(deri, fct):
        
        """Create Jacobian function,
        either numerically or from user-supplied function
        """
        
        if deri is None:
            return lambda x: nd.Jacobian(fct)(x) # Numerical differentiation
        elif isinstance(deri, np.ndarray):
            return lambda x: deri # Turn constant matrix into function
        else:
            return deri # Just use the supplied derivative function
