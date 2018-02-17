import numpy as np
from scipy.stats import norm
from GPy.kern import Exponential, Matern32, Matern52
from GPy.models import GPRegression
from scipy.optimize import minimize, basinhopping
from time import time


def expected_improvement(x, xi, fmax, gpreg, dim):
    pred = gpreg.predict(x.reshape(-1, dim))
    mean, std = pred[0].T[0], np.sqrt(pred[1].T[0])
    z = (mean - fmax - xi) / std
    ei = - std * (z * norm.cdf(z) + norm.pdf(z))
    ei[std == 0.0] = 0.0
    return ei


def upper_confidence_bound(x, beta, gpreg, dim):
    pred = gpreg.predict(x.reshape(-1, dim))
    mean, std = pred[0].T[0], np.sqrt(pred[1].T[0])
    return - mean - beta * std


class BO(object):
    def __init__(self, objective, bounds, n_iterations, n_init=2,
                 rand=None, kernel_function="Exponential",
                 acquisition_func="EI",
                 noise_var=0.001, log_info=True,
                 n_iters_aqui=15, use_bashinhopping=False):
        """
        Peter Kostovcik: Bayesian Optimization; diploma thesis 2017
        Charles University, faculty of mathematics and physics
        e-mail: p.kostovcik@gmail.com
        ======== INPUT ========
        objective:          objective function (input is numpy.array length = dimension)
        bounds:             box bounds (list of tuples)
        n_iterations:       number of iterations = evaluations of objective
        n_init:             number of starting points (default = 2)
        kernel_function     Exponential or Matern from GPy package
                            ["Exponential", "Matern32", "Matern52"]
                            (default = "Exponential")
        acquisition_func:   acquisition function ["UCB" == Upper Confidence Bound,
                                                 "EI" == Expected Improvement]
                                                 (default = EI)
        rand:               np.RandomState(some_number) (default = None, random choice)
        noise_var:          variance for noise (default = 0.001)
        log_info:           True/False -  (default = True)
        n_iters_aqui:       # restarts in optimization of acquisition function (default = 15)
        use_bashinhopping   True/False - use Bashinhopping algorithm (default = False)
        =================

        """
        self.objective, self.bounds = objective, bounds
        self.n_iterations, self.n_init = n_iterations, n_init
        self.kernel_function = kernel_function
        self.acquisition_func = acquisition_func
        self.noise_var = noise_var
        self.bashop = use_bashinhopping
        if rand is None:
            self.rand = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rand = rand
        self.log_info = log_info
        self.dim, self.n_iters_aqui = len(self.bounds), n_iters_aqui
        if self.kernel_function == "Exponential":
            self.kernel = Exponential(input_dim=self.dim)
        elif self.kernel_function == "Matern52":
            self.kernel = Matern52(input_dim=self.dim)
        else:
            self.kernel = Matern32(input_dim=self.dim)
        self.computation_time = None
        self.run()

    def run(self):
        self.start()
        self.maximize(self.n_iterations - self.n_init)
        self.result()

    def start(self):
        self.start_time1 = time()
        x_init = self.rand.uniform(low=np.array(self.bounds).T[0],
                                   high=np.array(self.bounds).T[1], size=(2, self.dim))
        y_init = np.array([self.objective(x_init[0]), self.objective(x_init[1])])
        self.x, self.y = [x_init[0], x_init[1]], [y_init[0], y_init[1]]
        if self.log_info:
            print("%s step in BO; objective value: %s at %s; time: %s s." % (
            1, self.y[0], self.x[0], time() - self.start_time1))
            print("%s step in BO; objective value: %s at %s; time: %s s." % (
            2, self.y[1], self.x[1], time() - self.start_time1))
        self.aquis_par = [np.abs(y_init[0] - y_init[1]) / 2 if self.acquisition_func == "EI" else 3]
        self.a = 2
        self.end_time1 = time()

    def maximize(self, iterations):
        self.start_time2 = time()
        for n in np.arange(iterations):
            X = np.array(self.x).reshape(-1, 1) if self.dim == 1 else np.array(self.x)
            Y = np.array(self.y).reshape(-1, 1)
            gpreg = GPRegression(X, Y, kernel=self.kernel, noise_var=self.noise_var)
            gpreg.optimize()  # MAP for kernel hyper-parameters

            vals, par = self.minimize_negative_acquisition(gpreg)

            if self.dim == 1:
                self.give_new_point(vals, par)
            else:
                self.x.append(par[vals.argmin()])
            self.y.append(self.objective(self.x[-1]))
            if self.log_info:
                print("%s step in BO; objective value: %.4f at %.4f; time: %.2f s." %
                      (len(self.x), self.y[-1], self.x[-1],
                       time() - self.start_time2 + self.end_time1 - self.start_time1))
            if n % 10 == 0:
                self.a += 1
            self.aquis_par.append(self.sample_aquis_param(self.a))
        self.end_time2 = time()

    def minimize_negative_acquisition(self, gpreg):
        # minimization of negative acquisition function
        vals, par = [], []
        x0 = list(self.rand.uniform(np.array(self.bounds).T[0], np.array(self.bounds).T[1],
                                    size=(self.n_iters_aqui - 1, self.dim)
                                    )) + [self.x[int(np.argmax(self.y))]]
        for x in x0:
            if self.acquisition_func == "EI":
                if self.bashop:
                    opti = basinhopping(expected_improvement, x0=x,
                                        minimizer_kwargs={"method": "L-BFGS-B",
                                                          "bounds": self.bounds,
                                                          "args": (self.aquis_par[-1],
                                                                   np.max(self.y),
                                                                   gpreg, self.dim,)})
                else:
                    opti = minimize(expected_improvement, x0=x, method="L-BFGS-B",
                                    args=(self.aquis_par[-1], np.max(self.y),
                                          gpreg, self.dim,),
                                    bounds=self.bounds)
            else:
                if self.bashop:
                    opti = basinhopping(upper_confidence_bound, x0=x,
                                        minimizer_kwargs={"method": "L-BFGS-B",
                                                          "bounds": self.bounds,
                                                          "args": (self.aquis_par[-1],
                                                                   gpreg, self.dim,)})
                else:
                    opti = minimize(upper_confidence_bound, x0=x, method="L-BFGS-B",
                                    args=(self.aquis_par[-1], gpreg, self.dim,),
                                    bounds=self.bounds)
            par.append(opti.x)
            vals.append(opti.fun)
        return np.array(vals), np.array(par)

    def give_new_point(self, values, parameters):
        if parameters[values.argmin()] in self.x and len(parameters) > 1:
            p, v = list(parameters), list(values)
            p.remove(parameters[values.argmin()])
            v.remove(values.min())
            self.give_new_point(np.array(v), np.array(p))
        else:
            self.x.append(parameters[values.argmin()])

    def sample_aquis_param(self, a):
        y = np.sort(self.y)
        if self.acquisition_func == "EI":
            return self.rand.uniform(0, 1 / a * (y[-1] - y[0]))
        else:
            return self.rand.uniform(0, 6 / a)

    def result(self):
        self.optimum = np.max(self.y)
        self.optimal_x = self.x[int(np.argmax(self.y))]
        if self.computation_time is None:
            self.computation_time = self.end_time1 - self.start_time1 + self.end_time2 - self.start_time2
        else:
            self.computation_time += self.end_time2 - self.start_time2
        if self.log_info:
            print("===================================================")
        print("Found max value %.4f at point %s. Computation time %.2f s." %
              (self.optimum, np.round(self.optimal_x, 4), self.computation_time))


def REMBO(f, D, d, n_iterations, n_init=2, rand=None,
          kernel_function="Exponential", acquisition_func="EI",
          noise_var=0.001, log_info=True, n_iters_aqui=15, use_bashinhopping=False):
    """
    ===========    REMBO = Random Embedding Bayesian Optimization    ===========
    is used for high dimension objective functions where not all dimensions are
    effective (active) - non-effective dimensions have no impact on function value
    or small impact (something like small random noise).
    More independent runs are needed!
    ======== INPUT ========
    f          objective function
    D          true dimension
    d > 1      (d >= d_e) lower dimension
    other inputs are same as for BO
    """
    if rand is None:
        rand = np.random.RandomState(np.random.randint(0, 10000))
    A = rand.normal(size=(D, d))

    def rho(z):
        x = A.dot(z)
        x[x < -1] = -1
        x[x > 1] = 1
        return x

    def objective_reduced(z):
        return f(rho(z))

    bounds = d * [(-np.sqrt(d), np.sqrt(d))] if d > 2 else d * [(-np.sqrt(d) / np.log(d), np.sqrt(d) / np.log(d))]

    return [BO(objective_reduced, bounds, n_iterations, n_init, rand, kernel_function,
               acquisition_func, noise_var, log_info, n_iters_aqui, use_bashinhopping), A]


if __name__ == "__main__":
    print("Simple examples of BO and REMBO are in test.py")
