import numpy as np
import numdifftools as nd
import scipy.optimize as opt
import cvxopt as cvx
from statsmodels.regression.quantile_regression import QuantReg
from scipy.stats import chi2, norm
from scipy.linalg import block_diag, null_space


"""Worst-case standard errors for minimum distance estimates
without knowledge of the full correlation matrix for the matched moments

If desired, also computes:
- worst-case efficient estimates
- full-information efficient estimates
- joint test of parameter restrictions
- over-identification test

Reference:
Cocci, Matthew D. & Mikkel Plagborg-Moller, "Standard Errors for Calibrated Parameters"
https://scholar.princeton.edu/mikkelpm/calibration
"""


class MinDist:
    
    
    def __init__(self, moment_fct, moment_estim,
                 moment_se=None, moment_varcov=None, moment_fct_deriv=None):
        
        self.moment_fct = moment_fct
        self.moment_estim = np.asarray(moment_estim).flatten()
        self.moment_num = len(moment_estim)
        self.moment_fct_deriv = self._deriv(moment_fct_deriv, self.moment_fct)
        
        # Var-cov matrix
        assert (moment_se is None) + (moment_varcov is None) == 1, 'Either "moment_se" or "moment_varcov" must be supplied, but not both'
        if moment_varcov is not None:
            self.moment_varcov = np.asarray(moment_varcov)
        else:
            self.moment_varcov = np.empty((self.moment_num,self.moment_num))
            self.moment_varcov[np.eye(self.moment_num)==1] = np.asarray(moment_se**2)
            self.moment_varcov[np.eye(self.moment_num)==0] = np.nan
        
        # Check inputs
        assert self.moment_varcov.shape==(self.moment_num,self.moment_num), 'Dimension of "moment_se" or "moment_varcov" is wrong'
        assert np.isreal(self.moment_estim).all(), 'Wrong input type for "moment_estim"'
        assert np.isreal(self.moment_varcov).all(), 'Wrong input type for "moment_se" or "moment_varcov"'
        assert (np.diag(self.moment_varcov)>=0).all(), 'SE for each individual moment must be nonnegative'
        assert np.any([self.moment_varcov==self.moment_varcov.T, np.isnan(self.moment_varcov)],axis=0).all(), '"moment_varcov" must be symmetric'
        assert self.moment_num>=1, '"moment_estim" must have at least one element'
        
        # Determine type of var-cov matrix
        self.full_info = (not np.isnan(self.moment_varcov).any()) # Full info
        self.diag_only = ((not self.full_info) and np.isnan(self.moment_varcov[np.eye(self.moment_num)==0]).all()) # Only diagonal is known
        
        # Check if var-cov has known block diagonal (and unknown everywhere else)
        self.blockdiag_only = False
        if not (self.full_info or self.diag_only):
            i = 0
            block_inds = []
            while i<self.moment_num: # Loop through purported blocks
                the_block = i + np.flatnonzero(1-np.isnan(self.moment_varcov[i,i:]))
                if not (np.diff(the_block)==1).all():
                    return # Can't be block diagonal
                block_inds.append(the_block)
                i = the_block.max()+1
            # Check that the block diagonal indeed has non-NaN values, with NaN's outside blocks
            block_bool = block_diag(*[np.ones((len(b),len(b))) for b in block_inds])
            self.blockdiag_only = (not np.isnan(self.moment_varcov[block_bool==1]).any()) and np.isnan(self.moment_varcov[block_bool==0]).all()
            if not self.blockdiag_only:
                return
        
            # If block diagonal, extract useful information for later
            self.moment_varcov_blocks = {'ind': block_inds, # Indices of blocks
                                         'varcov': [self.moment_varcov[np.ix_(b,b)] for b in block_inds], # Var-cov for each block
                                         'num': len(block_inds)}
            self.moment_varcov_blocks['chol'] = [(b if b.ndim==1 else np.linalg.cholesky(b)) for b in self.moment_varcov_blocks['varcov']] # Cholesky factors
            
    
    def fit(self, transf=lambda x: x, weight_mat=None,
            opt_init=None, estim_fct=None, eff=True, one_step=True, transf_deriv=None,
            param_estim=None, estim=None, transf_jacob=None, moment_fit=None, moment_jacob=None):
        
        """Minimum distance estimates and standard errors,
        either with full-information moment var-cov matrix
        or with limited-information individual moment variances
        """
        
        # Check inputs
        assert (param_estim is not None) or (opt_init is not None) or (estim_fct is not None), 'One of the following must be supplied: "param_estim", "opt_init", or "estim_fct"'
        
        # Transformation Jacobian function
        transf_deriv = self._deriv(transf_deriv, transf)
        
        # Determine weight matrix, if not supplied
        if self.full_info and (eff or (weight_mat is None)):
            weight_mat = np.linalg.inv(self.moment_varcov) # Full-info efficient weight matrix
        if weight_mat is None:
            weight_mat = np.diag(1/np.diag(self.moment_varcov)) # Ad hoc diagonal weight matrix
        
        # Default estimation routine
        if estim_fct is None:
            estim_fct = lambda W: opt.minimize(lambda x: (self.moment_estim-self.moment_fct(x)) @ W @ (self.moment_estim-self.moment_fct(x)), opt_init, method='BFGS')['x']
        
        # Initial estimate of parameters, if not supplied
        if param_estim is None:
            param_estim = estim_fct(weight_mat)
        
        # Transformation, moment function, and Jacobians at initial estimate
        estim, transf_jacob, moment_fit, moment_jacob \
            = self.estim_update(param_estim, transf, transf_deriv,
                                estim=estim, transf_jacob=transf_jacob,
                                moment_fit=moment_fit, moment_jacob=moment_jacob)
        moment_loadings = self._get_moment_loadings(moment_jacob, weight_mat, transf_jacob)
        estim_num = 1 if np.isscalar(estim) else len(estim)
        
        # Efficient estimates
        if eff:
            
            if self.full_info: # Full information
            
                if one_step: # One-step estimation
                    param_estim = self._get_onestep(moment_fit, weight_mat, moment_jacob, param_estim)
                    estim, transf_jacob, moment_fit, moment_jacob = self.estim_update(param_estim, transf, transf_deriv)
                else: # Full optimization
                    # Do nothing, since param_estim already contains estimates of interest
                    pass
                
            else: # Limited information
            
                if estim_num > 1: # If more than one parameter of interest, handle each separately by recursive call
                
                    estim_init = estim.copy()
                    ress = [self.fit(transf=lambda x: transf(x)[i],
                                           weight_mat=weight_mat,
                                           estim_fct=estim_fct,
                                           eff=True, one_step=one_step,
                                           param_estim=param_estim,
                                           estim=estim_init[i],
                                           transf_jacob=transf_jacob[i,:],
                                           moment_fit=moment_fit,
                                           moment_jacob=moment_jacob)
                            for i in range(estim_num)] # Compute for each parameter of interest
                    estim = np.array([r['estim'] for r in ress])
                    estim_se = np.array([r['estim_se'] for r in ress])
                    moment_loadings = np.array([r['moment_loadings'] for r in ress]).T
                    weight_mat = [r['weight_mat'] for r in ress]
                    
                else: # If only single parameter of interest
                
                    estim_se, moment_loadings, weight_mat = self.worstcase_eff(moment_jacob, transf_jacob, weight_mat=weight_mat)
                    if one_step: # One-step estimation
                        estim = self._get_onestep(moment_fit, None, moment_loadings, estim).item()
                    else: # Full optimization estimation
                        param_estim = estim_fct(weight_mat)
                        estim = transf(param_estim)
        
        # Start building results dictionary
        res = {'estim': estim,
               'param_estim': param_estim,
               'weight_mat': weight_mat,
               'moment_fit': moment_fit,
               'moment_jacob': moment_jacob,
               'moment_loadings': moment_loadings,
               'transf_jacob': transf_jacob,
               'estim_num': estim_num}
        
        # Standard errors
        if self.full_info: # Full information
            estim_varcov = moment_loadings.T @ self.moment_varcov @ moment_loadings
            estim_se = np.sqrt(np.diag(estim_varcov))
            res['estim_varcov'] = estim_varcov
        else: # Limited information
            if eff:
                # Do nothing, since standard errors have already been computed above
                pass
            else:
                estim_se, worstcase_varcov = self.worstcase_se(moment_loadings)
                res['worstcase_varcov'] = worstcase_varcov
        
        res['estim_se'] = estim_se
        
        return res
    
    
    def test(self, estim_res, joint=True, test_weight_mat=None):
        
        """Test whether transformed parameters equal zero
        """
        
        # t-statistics
        old_settings=np.seterr(divide='ignore')
        tstat = estim_res['estim']/estim_res['estim_se']
        np.seterr(**old_settings)
        tstat_pval = 2*norm.cdf(-np.abs(tstat))
        res = {'tstat': tstat,
               'tstat_pval': tstat_pval}
        
        if not joint:
            return res
        
        # Weight matrix for joint test statistic
        if test_weight_mat is None:
            if self.full_info: # Full information
                test_weight_mat = np.linalg.inv(estim_res['estim_varcov'])
            else: # Limited information
                # Ad hoc choice motivated by independence
                test_weight_mat = np.linalg.inv(estim_res['moment_loadings'].T @ np.diag(np.diag(self.moment_varcov)) @ estim_res['moment_loadings'])
        
        # Check dimensions
        assert test_weight_mat.shape == (estim_res['estim_num'],estim_res['estim_num']), 'Dimension of "test_weight_mat" is wrong'
        
        # Test statistic
        joint_stat = estim_res['estim'] @ test_weight_mat @ estim_res['estim']
        
        # p-value
        if self.full_info:
            joint_pval = 1-chi2.cdf(joint_stat, estim_res['estim_num'])
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
        
        assert isinstance(estim_res['weight_mat'], np.ndarray), 'Estimation results must be based on a single weight matrix'
        
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
                                 estim=moment_error,
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
               'moment_error_se': the_estim_res['estim_se'],
               'tstat': the_test_res['tstat'],
               'tstat_pval': the_test_res['tstat_pval']}
        
        if joint:
            res.update({'joint_stat': the_test_res['joint_stat'],
                        'joint_pval': the_test_res['joint_pval']})
            if self.full_info: # Adjust degrees of freedom
                res['joint_pval'] = 1-chi2.cdf(the_test_res['joint_stat'], self.moment_num-len(estim_res['param_estim']))
    
        return res
        
    
    def worstcase_se(self, moment_loadings):
        
        """Worst-case standard errors and corresponding var-cov matrix
        for linear combination of moments
        """
        
        if moment_loadings.ndim>1: # If more than one parameter of interest, handle them separately
            ress = [self.worstcase_se(moment_loadings[:,i]) for i in range(moment_loadings.shape[1])]
            return np.array([r[0] for r in ress]), [r[1] for r in ress]
        
        if self.diag_only: # Only diagonal is known
        
            moment_se = np.sqrt(np.diag(self.moment_varcov))
            se = moment_se @ np.abs(moment_loadings) # Closed form
            aux = np.sign(moment_loadings) * moment_se
            varcov = np.outer(aux, aux)
            
        elif self.blockdiag_only: # Only block diagonal is known
        
            loading_blocks = [moment_loadings[ind] for ind in self.moment_varcov_blocks['ind']]
            aux = [self.moment_varcov_blocks['varcov'][i] \
                   @ loading_blocks[i]
                   for i in range(self.moment_varcov_blocks['num'])]
            var_blocks = [max(loading_blocks[i] @ aux[i],1e-10) # Avoid exact zeros (when loadings are zero)
                          for i in range(self.moment_varcov_blocks['num'])]
            se = np.sqrt(var_blocks).sum()
            aux2 = [aux[i]/np.sqrt(var_blocks[i]) for i in range(self.moment_varcov_blocks['num'])]
            aux3 = [self.moment_varcov_blocks['chol'][i] \
                    - np.outer(aux[i], loading_blocks[i] @ self.moment_varcov_blocks['chol'][i]) \
                    / var_blocks[i]
                    for i in range(self.moment_varcov_blocks['num'])]
            aux4 = np.hstack((np.hstack(aux2).reshape(-1,1), block_diag(*aux3)))
            varcov = aux4 @ aux4.T
            
        else: # General knowledge of var-cov matrix
        
            # Solve semidefinite programming problem
            var, varcov = self.solve_sdp(np.outer(moment_loadings, moment_loadings))
            se = np.sqrt(var)
            varcov = np.array(varcov) # Convert to numpy array
            
        return se, varcov
    
    
    def worstcase_eff(self, moment_jacob, transf_jacob, weight_mat=None):
        
        """Compute worst-case efficient moment loadings and weight matrix
        See main paper for explanation
        """
        
        # Set up median regression as described in paper
        (p,k) = moment_jacob.shape
        GpG = moment_jacob.T @ moment_jacob
        Y = moment_jacob @ np.linalg.solve(GpG, transf_jacob.reshape(-1,1))
        moment_jacob_perp = null_space(moment_jacob.T)
        X = -moment_jacob_perp
        
        if self.diag_only: # Only diagonal is known
        
            # Run median regression
            moment_se = np.sqrt(np.diag(self.moment_varcov))
            qr_mod = QuantReg(moment_se.reshape(-1,1) * Y, moment_se.reshape(-1,1) * X)
            qr_fit = qr_mod.fit(q=.5)
            resid = qr_fit._results.resid # Residuals
            moment_loadings = resid / moment_se
            se = np.abs(resid).sum()
            
            # Weight matrix puts weight on only k moments
            sort_inds = np.abs(moment_loadings).argsort()
            weight_mat_new = weight_mat.copy()
            weight_mat_new[sort_inds[:p-k],:] = 0
            weight_mat_new[:,sort_inds[:p-k]] = 0
            
        else: # General case
            
            # Objective function and gradient
            def objective(z, Y, X):
                resid = Y.flatten() - X @ z
                se, varcov = self.worstcase_se(resid)
                grad = -2 * (X.T @ varcov @ resid)
                return se**2, grad
        
            # Solve nested optimization
            cvx.solvers.options['show_progress'] = False # Suppress CVX output temporarily
            opt_res = opt.minimize(lambda z: objective(z, Y, X), np.zeros(p-k), jac=True, method='BFGS')
            cvx.solvers.options['show_progress'] = True
            moment_loadings = Y.flatten() - X @ opt_res['x']
            se = np.sqrt(opt_res['fun'])
            
            # Weight matrix
            aux1 = np.outer(transf_jacob, opt_res['x']) / (transf_jacob @ np.linalg.solve(GpG, transf_jacob.reshape(-1,1)))
            W = lambda delta: np.vstack((np.hstack((np.eye(k),aux1)),np.hstack((aux1.T,delta*np.eye(p-k)))))
            # Determine delta such that W(delta) is positive definite
            delta_pd = opt.fsolve(lambda delta: np.linalg.eigvalsh(W(delta)).min()-0.01, 0)
            aux2 = np.hstack((moment_jacob, moment_jacob_perp))
            weight_mat_new = aux2 @ W(delta_pd) @ aux2.T
            
        return se, moment_loadings, weight_mat_new
    
    
    def solve_sdp(self, A):
        
        """Solve semidefinite programming problem
        max tr(A*V) s.t. V psd and known elements of V
        using CVXOPT package
        """
        
        # Find elements of V with known values (below diagonal)
        inds = np.all([(np.tril(np.ones((self.moment_num,self.moment_num)))==1),1-np.isnan(self.moment_varcov)],axis=0)
        
        # Coefficient matrices for CVXOPT: max tr(h*V) s.t. G'*vec(V)+c=0
        # Note: For some reason, CVXOPT's "vec" operator multiplies off-diagonal elements by 2
        factor = 2-np.eye(self.moment_num)
        c = cvx.matrix(-(self.moment_varcov*factor)[inds.T].reshape(-1,1),tc='d')
        G = cvx.sparse(cvx.matrix(np.eye(self.moment_num**2)[:,inds.T.flatten()==1]))
        h = cvx.matrix(-A,tc='d')
        
        # Solve SDP
        sol = cvx.solvers.sdp(c,Gs=[G],hs=[h])
        
        # Return objective value and optimal V
        return sol['dual objective'], sol['zs'][0]
    
    
    def estim_update(self, param_estim, transf, transf_deriv,
                     estim=None, transf_jacob=None, moment_fit=None, moment_jacob=None):
            
            """Update estimated parameter transformation and moments,
            including their Jacobians.
            Avoids recomputing quantities if they're already supplied.
            """
            
            if estim is None:
                estim = transf(param_estim)
            if transf_jacob is None:
                transf_jacob = transf_deriv(param_estim)
            if moment_fit is None:
                moment_fit = self.moment_fct(param_estim)
            if moment_jacob is None:
                moment_jacob = self.moment_fct_deriv(param_estim)
            return estim, transf_jacob, moment_fit, moment_jacob
    
    
    def _get_onestep(self, moment_init, weight_mat, moment_jacob, param_init):
        
        """One-step estimation
        """
        
        if weight_mat is None:
            subtr = moment_jacob.T @ (moment_init-self.moment_estim)
        else:
            subtr = np.linalg.solve(moment_jacob.T @ weight_mat @ moment_jacob, moment_jacob.T @ weight_mat @ (moment_init-self.moment_estim))
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
