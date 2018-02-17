# Bayesian optimization 

Bayesian Optimization (BO) is global optimization for expensive black-box functions. This is **light-weight implementation of Bayesian Optimization in python** using *numpy*, *scipy* and *GPy* created for author's [diploma thesis](https://is.cuni.cz/webapps/zzp/detail/185046/) (not in english, note that this code is updated).

This method doesn't optimize objective function directly. Instead of that it creates probabilistic model (Gaussian Process Regression) and uses it to calculate differentiable acquisition function which is cheap. Maximization of acquisition function is not easy problem, because of multimodality and/or flatness, but it is much easier than direct optimization of objective function. Maximization of acquisition functions gives us new point in which we evaluate objective function. We use calculated values and points as data for improving our probabilistic model in next iteration. Numerical optimisation in higher dimensions is still problem even for simpler functions, so we use dimensionality reduction via Random Embeddings.

In `bayesian_optimization.py` are implemented methods BO and REMBO. REMBO is used for high dimensions, when not all dimensions can change function values (mostly works good although we don't have effective dimensions - see references). 
In `test.py` are simple examples to show the usage of BO and REMBO. Any functions can be used, even those wich are not continuous in some finite set of points. 

BO is mostly used for expensive or not known functions. When dealing with known functions and there exists optimization method which fits good for that problem, use it. Use BO when you don't know what other method could be better (when you can use only random search - BO has better performance).

### Dependencies:

* numpy
* scipy
* GPy

### References
Main references used in diploma thesis (needed for better understanding of BO and REMBO algorithms):

[Taking the Human Out of the Loop: A Review of Bayesian Optimization](http://ieeexplore.ieee.org/document/7352306/references?part=1)

[Bayesian Optimization in a Billion Dimensions via Random Embeddings](https://arxiv.org/pdf/1301.1942.pdf)
