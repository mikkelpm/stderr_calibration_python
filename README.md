# Standard Errors for Calibrated Parameters

Python package that computes worst-case standard errors (SE) for minimum distance estimators, given knowledge of only the marginal variances (but not correlations) of the matched moments. The computed worst-case SE for the estimated parameters are sharp upper bounds on the true SE (which depend on the unknown moment correlation structure). For over-identified models, the package also computes the efficient moment selection that minimizes the worst-case SE. Additionally, the package can carry out tests of parameter restrictions or over-identifying restrictions.

**Reference:**
Cocci, Matthew D., and Mikkel Plagborg-Møller (2021), "Standard Errors for Calibrated Parameters", https://scholar.princeton.edu/mikkelpm/calibration

**Requirements:**
The Python packages [cvxopt](https://cvxopt.org/) and [numdifftools](https://pypi.org/project/numdifftools/) are required.

Tested in: Python 3.8.8 on Windows 10 PC

## Contents

- [example.ipynb](example.ipynb): Simple interactive example in Jupyter Notebook illustrating the main functionality of the package (also available in [HTML](docs/example.html) format)

- [stderr_calibration](stderr_calibration): Python package for minimum distance estimation, standard errors, and testing

- [tests](tests): Unit tests intended for use with the [pytest](https://docs.pytest.org/) framework

## Acknowledgements

This material is based upon work supported by the NSF under Grant #1851665. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.