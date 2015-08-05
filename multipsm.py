import numpy as np
import bsplines
import logging
import owlqn
from scipy import stats
from scipy import linalg
from scipy.misc import logsumexp


class MultiPSM:
    def __init__(self, prior, likelihoods, parameters=None, no_adjust={}, penalty=1e-3, seed=0):
        self.prior = prior
        self.likelihoods = likelihoods
        self.parameters = parameters
        self.no_adjust = {}
        self.penalty = penalty
        self.num_markers = len(likelihoods)
        self.num_subtypes = [lkl.num_subtypes for lkl in self.likelihoods]

    def posterior(self, example):
        loglik = []
        for i, lkl in enumerate(self.likelihoods):
            x, f, _, y = example[i]
            ll = lkl.log_proba(x, f, y, *self.parameters[i])
            loglik.append(ll)

        all_f = [f_sub for x, f_pop, f_sub, y in example]
        logl, marg, pair = self.prior.inference(all_f, loglik)

        return np.exp(marg[0])

    def predict(self, example, xnew):
        pz = self.posterior(example)
        w1, W2 = self.parameters[0]
        _, f, _, _ = example[0]
        Yhat = self.likelihoods[0].predict(xnew, f, w1, W2)
        yhat = np.dot(pz, Yhat)
        return yhat

    def fit(self, examples, max_iterations=100, tol=1e-5):
        assert len(examples[0]) == self.num_markers

        # Initialize parameters
        if self.parameters is None:
            parameters = [lkl.new_parameters() for lkl in self.likelihoods]

            by_marker = zip(*examples)
            for i, markers in enumerate(by_marker):
                y = np.concatenate([m[-1] for m in markers])
                q = np.linspace(0, 100, self.num_subtypes[i] + 2)
                p = np.percentile(y, q[1:-1])

                for j, yp in enumerate(p):
                    parameters[i][1][j, :] = yp
                    
        else:
            parameters = self.parameters

        total_logl = -float('inf')

        for itx in range(max_iterations):

            # E-step
            
            loglik = []
            for i, lkl in enumerate(self.likelihoods):
                ll = np.zeros((len(examples), self.num_subtypes[i]))
                for j, ex in enumerate(examples):
                    x, f, _, y = ex[i]
                    ll[j, :] = lkl.log_proba(x, f, y, *parameters[i])

                loglik.append(ll)

            all_logl = []
            all_marg = []
            all_pair = []
            for i, ex in enumerate(examples):
                all_f = [f_sub for x, f_pop, f_sub, y in ex]
                all_likel = [ll[i] for ll in loglik]
                logl, marg, pair = self.prior.inference(all_f, all_likel)
                all_logl.append(logl)
                all_marg.append(marg)
                all_pair.append(pair)

            old_logl = total_logl
            total_logl = sum(all_logl)
            delta = (total_logl - old_logl) / abs(total_logl)
            logging.info('Iteration:{:04d}, LL={:.10f}, dLL={:.10f}'.format(itx, total_logl, delta))
            if delta < tol:
                break
            
            # M-step

            ## First, optimize the prior with the expected observations.

            def prior_obj(w, examples=examples, all_marg=all_marg, all_pair=all_pair):
                self.prior.set_weights(w)
                v = 0.0
                n = 0

                for ex, marg, pair in zip(examples, all_marg, all_pair):
                    all_f = [f_sub for x, f_pop, f_sub, y in ex]
                    obs_m = [np.exp(m) for m in marg]
                    obs_p = [np.exp(p) for p in pair]
                    v -= self.prior.log_proba(obs_m, obs_p, all_f)
                    n += 1                    

                v /= n
                return v

            def prior_jac(w, examples=examples, all_marg=all_marg, all_pair=all_pair):
                self.prior.set_weights(w)
                g = np.zeros_like(w)
                n = 0

                for ex, marg, pair in zip(examples, all_marg, all_pair):
                    obs_m = [np.exp(m) for m in marg]
                    obs_p = [np.exp(p) for p in pair]
                    all_f = [f_sub for x, f_pop, f_sub, y in ex]

                    obs_f = []
                    obs_f += [outer(o, f).ravel() for o, f in zip(obs_m, all_f)]
                    obs_f += [o.ravel() for o in obs_p]
                    obs_f = np.concatenate(obs_f)

                    _, lp_marg, lp_pair = self.prior.inference(all_f)
                    exp_m = [np.exp(m) for m in lp_marg]
                    exp_p = [np.exp(p) for p in lp_pair]
                    
                    exp_f = []
                    exp_f += [outer(e, f).ravel() for e, f in zip(exp_m, all_f)]
                    exp_f += [e.ravel() for e in exp_p]
                    exp_f = np.concatenate(exp_f)

                    g += exp_f - obs_f
                    n += 1

                g /= n
                return g

            w0 = self.prior.get_weights()
            optim = owlqn.OWLQN(prior_obj, prior_jac, self.penalty, w0).minimize()
            self.prior.set_weights(optim.x)

            for k, likelihood in enumerate(self.likelihoods):
                if k in self.no_adjust:
                    continue
                
                examples_k = [ex[k] for ex in examples]
                marg_k = [np.exp(m[k]) for m in all_marg]
                w1, W2 = likelihood.fit(examples_k, marg_k)
                parameters[k] = (w1, W2)

            self.parameters = parameters


class MultiPSMPrior:
    def __init__(self, num_subtypes, num_sub_predictors, seed=0):
        self.num_subtypes = num_subtypes
        self.num_sub_predictors = num_sub_predictors
        self.target = 0

        rnd = np.random.RandomState(seed)
        
        singleton_weights_shapes = [(k, p) for k, p in zip(self.num_subtypes, self.num_sub_predictors)]
        self.singleton_weights = [rnd.normal(size=s) for s in singleton_weights_shapes]

        pair_weights_shapes = [(self.num_subtypes[self.target], k) for k in self.num_subtypes[(self.target + 1):]]
        self.pair_weights = [rnd.normal(size=s) for s in pair_weights_shapes]

    def log_proba(self, marg, pair, all_f):
        _, lp_marg, lp_pair = self.inference(all_f)
        ll = 0.0
        
        for p, lpp in zip(pair, lp_pair):
            ll += (p * lpp).sum()
            
        ll -= 2 * (marg[0] * lp_marg[0]).sum()
        
        return ll

    def single_features(self, all_f):
        encoded = []
        for i, f in enumerate(all_f):
            k = self.num_subtypes[i]
            z = np.ones(k)
            e = outer(z, f)
            encoded.append(e)

        return encoded

    def pair_features(self):
        encoded = []
        for i, w in enumerate(self.pair_weights):
            e = np.ones(w.shape)
            encoded.append(e)

        return encoded

    def inference(self, all_f, all_likel=None):
        use_likel = all_likel is not None

        # Initialize singleton clusters
        single_features = self.single_features(all_f)
        single_clusters = []
        for i, (w, e) in enumerate(zip(self.singleton_weights, single_features)):
            scores = (w * e).sum(axis=1)

            if use_likel:
                scores += all_likel[i]

            single_clusters.append(scores)

        # Initialize pair clusters
        pair_features = self.pair_features()
        pair_clusters = []
        for i, (w, e) in enumerate(zip(self.pair_weights, pair_features)):
            scores = w * e

            pair_clusters.append(scores)

        # Pass messages to root (target singleton cluster)
        for i, s in enumerate(single_clusters[1:]):
            pair_clusters[i] += rowvec(s)
            single_clusters[0] += marginalize(pair_clusters[i], [0], logsumexp)

        # Normalize root
        logl = logsumexp(single_clusters[0])
        single_clusters[0] -= logl

        # Pass messages from root
        for i, p in enumerate(pair_clusters, 1):
            p += colvec(single_clusters[0] - marginalize(p, [0], logsumexp))
            single_clusters[i] += marginalize(p, [1], logsumexp) - single_clusters[i]

        # Normalize non-root clusters
        for s in single_clusters[1:]:
            s -= logsumexp(s)

        for p in pair_clusters:
            p -= logsumexp(p)

        return logl, single_clusters, pair_clusters

    def set_weights(self, flat_weights):
        single_shapes = [w.shape for w in self.singleton_weights]
        pair_shapes = [w.shape for w in self.pair_weights]
        offset = 0

        for i, s in enumerate(single_shapes):
            n, m = s
            end = offset + (n * m)
            w = flat_weights[offset:end]
            self.singleton_weights[i] = w.reshape(s)
            offset += (n * m)

        for i, s in enumerate(pair_shapes):
            n, m = s
            end = offset + (n * m)
            w = flat_weights[offset:end]
            self.pair_weights[i] = w.reshape(s)
            offset += (n * m)

    def get_weights(self):
        single = [w.ravel() for w in self.singleton_weights]
        pair = [w.ravel() for w in self.pair_weights]
        return np.concatenate(single + pair)

    def _onehot_encode(self, z, k):
        e = np.zeros(k)
        e[z] = 1
        return e


class MultiPSMLikelihood:
    def __init__(self, num_subtypes, num_pop_predictors,
                 bs_lo, bs_hi, bs_degree, bs_dimension,
                 v_const, v_ou, l_ou, v_noise):

        self.num_subtypes = num_subtypes
        self.covariance_fn = Kernel(v_const, v_ou, l_ou, v_noise)
        self.pop_factor = LinearRegressionFactor(num_pop_predictors, self.covariance_fn)
        self.subpop_factor = BSplineMixtureFactor(num_subtypes, bs_lo, bs_hi, bs_degree, bs_dimension, self.covariance_fn)

    def new_parameters(self):
        w1 = np.zeros(self.pop_factor.num_features)
        W2 = np.array([np.zeros(n) for n in self.subpop_factor.num_features])
        return w1, W2

    def predict(self, x, f, w1, W2):
        pop_pred = self.pop_factor.predict(x, f, w1)
        sub_pred = [self.subpop_factor.predict(x, z, W2) for z in range(self.num_subtypes)]
        all_pred = [pop_pred + p for p in sub_pred]
        return np.array(all_pred)

    def log_proba(self, x, f, y, w1, W2):
        if len(x) == 0:
            return np.zeros(self.num_subtypes)
        
        pop_pred = self.pop_factor.predict(x, f, w1)
        sub_pred = [self.subpop_factor.predict(x, z, W2) for z in range(self.num_subtypes)]
        all_m = [pop_pred + p for p in sub_pred]
        S = self.covariance_fn(x)
        ll = [stats.multivariate_normal.logpdf(y, m, S) for m in all_m]
        return np.array(ll)

    def fit(self, examples, marg):
        w1, W2 = self.new_parameters()
        
        sub_suffstats = self.subpop_factor.new_suffstats()

        for i, (x, f_pop, f_sub, y) in enumerate(examples):
            if len(x) < 1:
                continue
            
            pz = marg[i]
            X = self.subpop_factor.feature_matrix(x)
            S = self.covariance_fn(x)
            
            yproj = self.pop_factor.projection_residuals(x, f_pop, y)            
            Xproj = self.pop_factor.projection_residuals(x, f_pop, X)

            ss1, ss2 = linreg_suffstats(Xproj, yproj, S)

            for j, p in enumerate(pz):
                sub_suffstats[j][0] += p * ss1
                sub_suffstats[j][1] += p * ss2

        for i, (ss1, ss2) in enumerate(sub_suffstats):
            W2[i] = linalg.solve(ss1, ss2)

        pop_suffstats = self.pop_factor.new_suffstats()

        for i, (x, f_pop, f_sub, y) in enumerate(examples):
            if len(x) < 1:
                continue
            
            pz = marg[i]
            X = self.pop_factor.feature_matrix(x, f_pop)
            S = self.covariance_fn(x)

            Yhat = np.array([self.subpop_factor.predict(x, z, W2) for z in range(self.num_subtypes)])
            yhat = np.dot(pz, Yhat)
            yres = y - yhat

            ss1, ss2 = linreg_suffstats(X, yres, S)
            pop_suffstats[0] += ss1
            pop_suffstats[1] += ss2

        w1[:] = linalg.solve(pop_suffstats[0], pop_suffstats[1])

        return w1, W2


class LinearRegressionFactor:
    def __init__(self, num_predictors, covariance_fn):
        self.num_predictors = num_predictors
        self.covariance_fn = covariance_fn

        def basis(x):
            x = x.ravel()
            n = x.size
            return colvec(np.ones(n))

        self.num_bases = 1
        self.basis = basis

    @property
    def num_features(self):
        return self.num_bases * self.num_predictors

    def new_suffstats(self, penalty=1e-1):
        n = self.num_features
        ss1 = penalty * np.eye(n)
        ss2 = np.zeros(n)
        return [ss1, ss2]

    def suffstats(self, x, f, y):
        X = self.feature_matrix(x, f)
        S = self.covariance_fn(x)
        return linreg_suffstats(X, y, S)

    def feature_matrix(self, x, f):
        n = x.size
        p = f.size
        f = rowvec(f)
        
        B = self.basis(x)
        X = np.zeros((n, B.shape[1] * p))
        for i, b in enumerate(B.T):
            i1 = i * p
            i2 = i1 + p
            X[:, i1:i2] = colvec(b) * f

        return X

    def project(self, x, f, y):
        X = self.feature_matrix(x, f)
        H = hat_matrix(X)

        S = self.covariance_fn(x)
        b = linalg.solve(S, y)
        
        return np.dot(H, b)

    def projection_residuals(self, x, f, y):
        yhat = self.project(x, f, y)
        return yhat - y

    def predict(self, x, f, w):
        X = self.feature_matrix(x, f)
        m = np.dot(X, w)
        return m

    def log_proba(self, x, f, y, w):
        m = self.predict(x, f, w)
        S = self.covariance_fn(x)
        return stats.multivariate_normal.logpdf(y, m, S)
    

class BSplineFactor:
    def __init__(self, lo, hi, degree, dimension, covariance_fn):
        bounds = (lo, hi)
        self.basis = bsplines.universal_basis(bounds, degree, dimension)
        self.weights = np.zeros(self.basis.dimension)
        self.covariance_fn = covariance_fn

    @property
    def num_features(self):
        return self.basis.dimension

    def new_suffstats(self, penalty=1e-1):
        n = self.num_features
        D = np.diff(np.eye(n))

        ss1 = penalty * np.dot(D, D.T)
        ss2 = np.zeros(n)

        return [ss1, ss2]

    def suffstats(self, x, y):
        X = self.feature_matrix(x)
        S = self.covariance_fn(x)
        return linreg_suffstats(X, y, S)

    def feature_matrix(self, x):
        return self.basis.eval(x)

    def predict(self, x, w):
        B = self.feature_matrix(x)
        m = np.dot(B, w)
        return m

    def log_proba(self, x, y, w):
        m = self.predict(x, w)
        S = self.covariance_fn(x)
        return stats.multivariate_normal.logpdf(y, m, S)


class BSplineMixtureFactor:
    def __init__(self, num_components, lo, hi, degree, dimension, covariance_fn):
        self.num_components = num_components
        self.bspline_factors = [BSplineFactor(lo, hi, degree, dimension, covariance_fn) for _ in range(self.num_components)]

    @property
    def num_features(self):
        return [f.num_features for f in self.bspline_factors]

    def new_suffstats(self, penalty=1e-1):
        ss = [f.new_suffstats(penalty) for f in self.bspline_factors]
        return ss

    def suffstats(self, x, y):
        factor = self.bspline_factors[0]
        return factor.suffstats(x, y)

    def feature_matrix(self, x):
        factor = self.bspline_factors[0]
        return factor.feature_matrix(x)

    def predict(self, x, z, W):
        factor = self.bspline_factors[z]
        return factor.predict(x, W[z])

    def log_proba(self, x, y, z, W):
        factor = self.bspline_factors[z]
        return factor.log_proba(x, y, W[z])


class Kernel:
    def __init__(self, v_const=1.0, v_ou=1.0, l_ou=1.0, v_noise=1.0):
        self.v_const = v_const
        self.v_ou = v_ou
        self.l_ou = l_ou
        self.v_noise = v_noise

    def __call__(self, x1, x2=None):
        symmetric = x2 is None
        d = differences(x1, x1) if symmetric else differences(x1, x2)
        
        K = self.v_const * np.ones_like(d)
        K += ou_kernel(d, self.v_ou, self.l_ou)
        if symmetric:
            K += self.v_noise * np.eye(x1.size)
            
        return K


def ou_kernel(d, v, l):
    return v * np.exp( - np.abs(d) / l )


def differences(x1, x2):
    return colvec(x1) - rowvec(x2)


def colvec(x):
    x = x.ravel()
    return x[:, np.newaxis]

def rowvec(x):
    return colvec(x).T


def outer(x, y):
    return colvec(x) * rowvec(y)


def hat_matrix(X, S=None, eps=1e-4):
    _, p = X.shape
    
    if S is not None:
        C = linalg.solve(S, X)
        C = np.dot(X.T, C) + eps * np.eye(p)
    else:
        C = np.dot(X.T, X) + eps * np.eye(p)

    A = linalg.solve(C, X.T)
    H = np.dot(X, A)
    
    return H


def linreg_suffstats(X, y, S=None):
    n = y.size
    
    if S is None:
        S = np.eye(n)

    ss1 = np.dot(X.T, linalg.solve(S, X))
    ss2 = np.dot(X.T, linalg.solve(S, y))

    return ss1, ss2


def marginalize(x, remove, func=np.sum):
    axes = tuple(range(x.ndim))
    over = set(axes) - set(remove)
    over = tuple(sorted(list(over)))
    return func(x, axis=over)
