# Standard Errors for Calibrated Parameters

Python package that computes worst-case standard errors (SE) for minimum distance estimators, given knowledge of only the marginal variances (but not correlations) of the matched moments.

The computed worst-case SE for the estimated parameters are sharp upper bounds on the true SE (which depend on the unknown moment correlation structure). For over-identified models, the package also computes the efficient moment selection that minimizes the worst-case SE. Additionally, the package can carry out tests of parameter restrictions or over-identifying restrictions.

**Reference:**
Cocci, Matthew D., and Mikkel Plagborg-Møller (2021), "Standard Errors for Calibrated Parameters", https://scholar.princeton.edu/mikkelpm/calibration

Tested in: Python 3.8.11 (Anaconda distribution) on Windows 10 PC

Other versions: [Matlab](https://github.com/mikkelpm/stderr_calibration_matlab)

## Contents

- [example.ipynb](example.ipynb): Simple interactive example in Jupyter Notebook illustrating the main functionality of the package (also available in [HTML format](https://mikkelpm.github.io/stderr_calibration_python/example.html))

- [example_ngm.ipynb](example_ngm.ipynb): Even simpler example in Jupyter Notebook in the context of the Neoclassical Growth Model (also available in [HTML format](https://mikkelpm.github.io/stderr_calibration_python/example_ngm.html))

- [stderr_calibration](stderr_calibration): Python package for minimum distance estimation, standard errors, and testing

- [estimate_hank.py](estimate_hank.py): Empirical application to estimation of a heterogeneous agent New Keynesian macro model, using impulse response estimates from [Chang, Chen & Schorfheide (2021)](https://cpb-us-w2.wpmucdn.com/web.sas.upenn.edu/dist/e/242/files/2021/05/EvalHAmodels_v6_pub.pdf) and [Miranda-Agrippino & Ricco (2021)](https://doi.org/10.1257/mac.20180124), which are stored in the [data](data) folder

- [sequence_jacobian](sequence_jacobian): Copy of the [Sequence-Space Jacobian](https://github.com/shade-econ/sequence-jacobian) package developed by [Auclert, Bardóczy, Rognlie & Straub (2021)](http://web.stanford.edu/~aauclert/sequence_space_jacobian.pdf), with minor changes made to the file [hank.py](sequence_jacobian/hank.py)

- [tests](tests): Unit tests intended for use with the [pytest](https://docs.pytest.org/) framework

## Requirements

The Python packages [cvxopt](https://cvxopt.org/) and [numdifftools](https://pypi.org/project/numdifftools/) are required.

## Acknowledgements

We thank [Minsu Chang](http://minsuchang.com) and [Silvia Miranda-Agrippino](http://silviamirandaagrippino.com) for supplying the moments used in the empirical application.

This material is based upon work supported by the NSF under Grant #1851665. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.
