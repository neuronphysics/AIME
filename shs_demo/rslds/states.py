import numpy as np
from scipy.misc import logsumexp

from pybasicbayes.util.stats import sample_discrete
from pyhsmm.internals.hmm_states import HMMStatesEigen

from pyslds.states import _SLDSStatesCountData, _SLDSStatesMaskedData

from rslds.util import one_hot, logistic

class InputHMMStates(HMMStatesEigen):

    def __init__(self, covariates, *args, **kwargs):
        self.covariates = covariates
        super(InputHMMStates, self).__init__(*args, **kwargs)

    @property
    def trans_matrix(self):
        return self.model.trans_distn.get_trans_matrices(self.covariates)

    def generate_states(self, initial_condition=None, with_noise=True, stateseq=None):
        if stateseq is None:
            As = self.trans_matrix
            self.stateseq = -1 * np.ones(self.T, dtype=np.int32)
            self.stateseq[0] = np.random.choice(self.num_states)
            for t in range(1, self.T):
                self.stateseq[t] = sample_discrete(As[t-1, self.stateseq[t-1], :].ravel())

        else:
            assert stateseq.shape == (self.T,)
            self.stateseq = stateseq.astype(np.int32)

class _RecurrentSLDSStatesBase(object):
    def __init__(self, model, covariates=None, data=None, **kwargs):

        if covariates is not None:
            raise NotImplementedError("Not supporting exogenous inputs yet")

        super(_RecurrentSLDSStatesBase, self).\
            __init__(model, data=data, **kwargs)

        self.covariates = self.gaussian_states[:-1]

    @property
    def trans_distn(self):
        return self.model.trans_distn

    def generate_states(self, initial_condition=None, with_noise=True, stateseq=None):
        from pybasicbayes.util.stats import sample_discrete
        T, K, n = self.T, self.num_states, self.D_latent

        dss = -1 * np.ones(T, dtype=np.int32) if stateseq is None else stateseq
        gss = np.empty((T,n), dtype='double')

        if initial_condition is None:
            init_state_distn = np.ones(self.num_states) / float(self.num_states)
            dss[0] = sample_discrete(init_state_distn.ravel())
            gss[0] = self.init_dynamics_distns[dss[0]].rvs()
        else:
            dss[0] = initial_condition[0]
            gss[0] = initial_condition[1]

        for t in range(1,T):
            A = self.trans_distn.get_trans_matrices(gss[t-1:t])[0]
            if with_noise:
                if dss[t] == -1:
                    dss[t] = sample_discrete(A[dss[t-1], :])

                gss[t] = self.dynamics_distns[dss[t-1]].\
                    rvs(x=np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])),
                        return_xy=False)
            else:
                if dss[t] == -1:
                    dss[t] = np.argmax(A[dss[t-1], :])

                gss[t] = self.dynamics_distns[dss[t-1]]. \
                    predict(np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])))
            assert np.all(np.isfinite(gss[t])), "SLDS appears to be unstable!"

        self.stateseq = dss
        self.gaussian_states = gss


class PGRecurrentSLDSStates(_RecurrentSLDSStatesBase,
                            _SLDSStatesCountData,
                            InputHMMStates):
    def __init__(self, model, covariates=None, data=None, mask=None,
                 stateseq=None, gaussian_states=None, **kwargs):

        super(PGRecurrentSLDSStates, self).\
            __init__(model, covariates=covariates, data=data, mask=mask,
                     stateseq=stateseq, gaussian_states=gaussian_states,
                     **kwargs)

        if not hasattr(self, 'ppgs'):
            import pypolyagamma as ppg

            num_threads = ppg.get_omp_num_threads()
            seeds = np.random.randint(2 ** 16, size=num_threads)
            self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

        self.trans_omegas = np.ones((self.T-1, self.num_states-1))

        if stateseq is not None and gaussian_states is not None:
            self.resample_transition_auxiliary_variables()

    @property
    def info_emission_params(self):
        J_node, h_node, log_Z_node = super(PGRecurrentSLDSStates, self).info_emission_params
        J_node_trans, h_node_trans = self.info_trans_params
        J_node[:-1] += J_node_trans
        h_node[:-1] += h_node_trans
        return J_node, h_node, log_Z_node

    @property
    def info_trans_params(self):
        trans_distn, omega = self.trans_distn, self.trans_omegas

        prev_state = one_hot(self.stateseq[:-1], self.num_states)
        next_state = one_hot(self.stateseq[1:], self.num_states)

        A = trans_distn.A[:, :self.num_states]
        C = trans_distn.A[:, self.num_states:self.num_states+self.D_latent]
        b = trans_distn.b

        CCT = np.array([np.outer(cp, cp) for cp in C]). \
            reshape((trans_distn.D_out, self.D_latent ** 2))
        J_node = np.dot(omega, CCT)

        kappa = trans_distn.kappa_func(next_state[:,:-1])
        h_node = kappa.dot(C)
        h_node -= (omega * b.T).dot(C)
        h_node -= (omega * prev_state.dot(A.T)).dot(C)

        J_node = J_node.reshape((self.T-1, self.D_latent, self.D_latent))
        return J_node, h_node

    def resample(self, niter=1):
        super(PGRecurrentSLDSStates, self).resample(niter=niter)
        self.resample_transition_auxiliary_variables()

    def resample_gaussian_states(self):
        super(PGRecurrentSLDSStates, self).resample_gaussian_states()
        self.covariates = self.gaussian_states[:-1].copy()

    def resample_transition_auxiliary_variables(self):
        trans_distn = self.trans_distn
        prev_state = one_hot(self.stateseq[:-1], self.num_states)
        next_state = one_hot(self.stateseq[1:], self.num_states)

        A = trans_distn.A[:, :self.num_states]
        C = trans_distn.A[:, self.num_states:self.num_states + self.D_latent]
        b = trans_distn.b

        psi = prev_state.dot(A.T) \
              + self.covariates.dot(C.T) \
              + b.T \

        b_pg = trans_distn.b_func(next_state[:,:-1])

        import pypolyagamma as ppg
        ppg.pgdrawvpar(self.ppgs, b_pg.ravel(), psi.ravel(), self.trans_omegas.ravel())


class _SoftmaxRecurrentSLDSStatesBase(_RecurrentSLDSStatesBase,
                                      _SLDSStatesMaskedData,
                                      InputHMMStates):
    def __init__(self, model, **kwargs):
        super(_SoftmaxRecurrentSLDSStatesBase, self).__init__(model, **kwargs)
        self.a = np.zeros((self.T - 1,))
        self.bs = np.ones((self.T - 1, self.num_states))

    @property
    def lambda_bs(self):
        return 0.5 / self.bs * (logistic(self.bs) - 0.5)

    def _set_expected_trans_stats(self):
        T, D, K = self.T, self.D_latent, self.num_states

        E_z = self.expected_states
        E_z_zp1T = self.expected_joints
        E_x = self.smoothed_mus
        E_x_xT = self.smoothed_sigmas + E_x[:, :, None] * E_x[:, None, :]

        E_u = np.concatenate((E_z[:-1], E_x[:-1]), axis=1)

        E_x_zp1T = E_x[:-1, :, None] * E_z[1:, None, :]
        E_u_zp1T = np.concatenate((E_z_zp1T, E_x_zp1T), axis=1)

        E_u_uT = np.zeros((T - 1, K + D, K + D))
        E_u_uT[:, np.arange(K), np.arange(K)] = E_z[:-1]
        E_u_uT[:, :K, K:] = E_z[:-1, :, None] * E_x[:-1, None, :]
        E_u_uT[:, K:, :K] = E_x[:-1, :, None] * E_z[:-1, None, :]
        E_u_uT[:, K:, K:] = E_x_xT[:-1]

        self.E_trans_stats = (E_u_zp1T, E_u_uT, E_u, self.a, self.lambda_bs)


class _SoftmaxRecurrentSLDSStatesMeanField(_SoftmaxRecurrentSLDSStatesBase):

    @property
    def expected_info_rec_params(self):
        E_z = self.expected_states
        E_W = self.trans_distn.expected_W
        E_WWT = self.trans_distn.expected_WWT
        E_logpi_WT = self.trans_distn.expected_logpi_WT

        J_rec = np.zeros((self.T, self.D_latent, self.D_latent))
        np.einsum('tk, kij -> tij', 2 * self.lambda_bs, E_WWT, out=J_rec[:-1])

        h_rec = np.zeros((self.T, self.D_latent))
        h_rec[:-1] += E_z[1:].dot(E_W.T)
        h_rec[:-1] += -1 * (0.5 - 2 * self.a[:,None] * self.lambda_bs).dot(E_W.T)
        h_rec[:-1] += -2 * np.einsum('ti, tj, jid -> td', E_z[:-1], self.lambda_bs, E_logpi_WT)

        return J_rec, h_rec

    @property
    def expected_info_emission_params(self):
        J_node, h_node, log_Z_node = \
            super(_SoftmaxRecurrentSLDSStatesMeanField, self).\
                expected_info_emission_params

        J_rec, h_rec = self.expected_info_rec_params
        return J_node + J_rec, h_node + h_rec, log_Z_node

    @property
    def mf_aBl(self):
        aBl = super(_SoftmaxRecurrentSLDSStatesMeanField, self).mf_aBl
        aBl += self._mf_aBl_rec
        return aBl

    @property
    def _mf_aBl_rec(self):
        aBl = np.zeros((self.T, self.num_states))

        E_x = self.smoothed_mus
        E_W = self.trans_distn.expected_W
        E_logpi = self.trans_distn.expected_logpi
        E_logpi_WT = self.trans_distn.expected_logpi_WT
        E_logpi_logpiT = self.trans_distn.expected_logpi_logpiT
        E_logpisq = np.array([np.diag(Pk) for Pk in E_logpi_logpiT]).T

        aBl[1:] += E_x[:-1].dot(E_W)

        aBl[:-1] += -2 * np.einsum('td, kid, tk -> ti', E_x[:-1], E_logpi_WT, self.lambda_bs)

        a, bs = self.a, self.bs
        aBl[:-1] += -1 * (0.5 - 2*a[:,None] * self.lambda_bs).dot(E_logpi.T)

        aBl[:-1] += -1 * self.lambda_bs.dot(E_logpisq.T)

        return aBl

    def meanfield_update_auxiliary_vars(self, n_iter=10):
        K = self.num_states
        E_z = self.expected_states
        E_z /= E_z.sum(1, keepdims=True)
        E_x = self.smoothed_mus
        E_xxT = self.smoothed_sigmas + E_x[:,:,None] * E_x[:,None,:]
        E_logpi = self.trans_distn.expected_logpi
        E_W = self.trans_distn.expected_W
        E_WWT = self.trans_distn.expected_WWT
        E_logpi_WT = self.trans_distn.expected_logpi_WT
        E_logpi_logpiT = self.trans_distn.expected_logpi_logpiT
        E_logpi_sq = np.array([np.diag(Pk) for Pk in E_logpi_logpiT]).T

        m = E_z[:-1].dot(E_logpi) + E_x[:-1].dot(E_W)

        psi_1 = E_z[:-1].dot(E_logpi_sq)

        psi_2 = 2 * np.einsum('td, ti, kid -> tk', E_x[:-1], E_z[:-1], E_logpi_WT)

        psi_3 = np.einsum('tij, kij -> tk', E_xxT[:-1], E_WWT)

        s = psi_1 + psi_2 + psi_3
        assert np.all(s > 0)
        for itr in range(n_iter):
            lambda_bs = self.lambda_bs

            self.a = 2 * (m * lambda_bs).sum(axis=1) + K / 2.0 - 1.0
            self.a /= 2 * lambda_bs.sum(axis=1)

            self.bs = np.sqrt(s - 2 * m * self.a[:, None] + self.a[:, None] ** 2)

    def meanfield_update_discrete_states(self):
        self.clear_caches()

        trans_potential = self.trans_distn.exp_expected_logpi
        init_potential = self.mf_pi_0
        likelihood_potential = self.mf_aBl
        alphal = self._messages_forwards_log(trans_potential, init_potential, likelihood_potential)
        betal = self._messages_backwards_log(trans_potential, likelihood_potential)

        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:, None]
        np.exp(expected_states, out=expected_states)
        expected_states /= expected_states.sum(1)[:, None]

        Al = np.log(trans_potential)
        log_joints = alphal[:-1, :, None] + betal[1:, None, :] \
                     + likelihood_potential[1:, None, :] \
                     + Al[None, ...]
        log_joints -= log_joints.max(axis=(1, 2), keepdims=True)
        joints = np.exp(log_joints)
        joints /= joints.sum(axis=(1, 2), keepdims=True)

        normalizer = logsumexp(alphal[0] + betal[0])

        self.expected_states = expected_states
        self.expected_joints = joints
        self.expected_transcounts = joints.sum(0)
        self._normalizer = normalizer

        self.stateseq = self.expected_states.argmax(1).astype('int32')

        self._mf_param_snapshot = \
            (np.log(trans_potential), np.log(init_potential),
             likelihood_potential, normalizer)

        from pyslds.util import hmm_entropy
        params = (np.log(trans_potential), np.log(init_potential), likelihood_potential, normalizer)
        stats = (expected_states, self.expected_transcounts, normalizer)
        return hmm_entropy(params, stats)

    def meanfieldupdate(self, niter=1):
        super(_SoftmaxRecurrentSLDSStatesMeanField, self).meanfieldupdate()
        self.meanfield_update_auxiliary_vars()
        self._set_expected_trans_stats()

    def get_vlb(self, most_recently_updated=False):
        from pyslds.util import expected_hmm_logprob

        vlb = expected_hmm_logprob(
            self.mf_pi_0, self.trans_distn.exp_expected_logpi,
            (self.expected_states, self.expected_transcounts, self._normalizer))

        vlb += np.sum(self.expected_states * self.mf_aBl)

        vlb += self._variational_entropy

        return vlb

    def _init_mf_from_gibbs(self):
        super(_SoftmaxRecurrentSLDSStatesBase, self)._init_mf_from_gibbs()
        self.meanfield_update_auxiliary_vars()
        self.expected_joints = self.expected_states[:-1, :, None] * self.expected_states[1:, None, :]
        self._mf_param_snapshot = \
            (self.trans_distn.expected_logpi, np.log(self.mf_pi_0),
             self.mf_aBl, self._normalizer)
        self._set_expected_trans_stats()


class _SoftmaxRecurrentSLDSStatesVBEM(_SoftmaxRecurrentSLDSStatesBase):
    def vb_E_step(self):
        H_z = self.vb_E_step_discrete_states()
        H_x = self.vb_E_step_gaussian_states()
        self.vbem_update_auxiliary_vars()
        self._set_expected_trans_stats()
        self._variational_entropy = H_z + H_x

    @property
    def vbem_info_rec_params(self):
        E_z = self.expected_states
        W = self.trans_distn.W
        WWT = np.array([np.outer(wk, wk) for wk in W.T])
        logpi = self.trans_distn.logpi
        logpi_WT = np.array([np.outer(lpk, wk) for lpk, wk in zip(logpi.T, W.T)])

        J_rec = np.zeros((self.T, self.D_latent, self.D_latent))
        np.einsum('tk, kij -> tij', 2 * self.lambda_bs, WWT, out=J_rec[:-1])

        h_rec = np.zeros((self.T, self.D_latent))
        h_rec[:-1] += E_z[1:].dot(W.T)
        h_rec[:-1] += -1 * (0.5 - 2 * self.a[:,None] * self.lambda_bs).dot(W.T)
        h_rec[:-1] += -2 * np.einsum('ti, tj, jid -> td', E_z[:-1], self.lambda_bs, logpi_WT)

        return J_rec, h_rec

    @property
    def vbem_info_emission_params(self):
        J_node, h_node, log_Z_node = \
            super(_SoftmaxRecurrentSLDSStatesVBEM, self). \
                vbem_info_emission_params

        J_rec, h_rec = self.vbem_info_rec_params
        return J_node + J_rec, h_node + h_rec, log_Z_node

    @property
    def vbem_aBl(self):
        aBl = super(_SoftmaxRecurrentSLDSStatesVBEM, self).vbem_aBl

        aBl += self._vbem_aBl_rec
        return aBl

    @property
    def _vbem_aBl_rec(self):
        aBl = np.zeros((self.T, self.num_states))

        E_x = self.smoothed_mus
        W = self.trans_distn.W
        logpi = self.trans_distn.logpi
        logpi_WT = np.array([np.outer(lpk, wk) for lpk, wk in zip(logpi.T, W.T)])
        logpisq = logpi**2

        aBl[1:] += E_x[:-1].dot(W)

        aBl[:-1] += -2 * np.einsum('td, kid, tk -> ti', E_x[:-1], logpi_WT, self.lambda_bs)

        a, bs = self.a, self.bs
        aBl[:-1] += -1 * (0.5 - 2*a[:,None] * self.lambda_bs).dot(logpi.T)

        aBl[:-1] += -1 * self.lambda_bs.dot(logpisq.T)

        return aBl

    def vbem_update_auxiliary_vars(self, n_iter=10):
        K = self.num_states
        E_z = self.expected_states
        E_z /= E_z.sum(1, keepdims=True)
        E_x = self.smoothed_mus
        E_xxT = self.smoothed_sigmas + E_x[:,:,None] * E_x[:,None,:]
        logpi = self.trans_distn.logpi
        W = self.trans_distn.W
        WWT = np.array([np.outer(wk, wk) for wk in W.T])
        logpi_WT = np.array([np.outer(lpk, wk) for lpk, wk in zip(logpi.T, W.T)])
        logpi_sq = logpi**2

        m = E_z[:-1].dot(logpi) + E_x[:-1].dot(W)

        psi_1 = E_z[:-1].dot(logpi_sq)

        psi_2 = 2 * np.einsum('td, ti, kid -> tk', E_x[:-1], E_z[:-1], logpi_WT)

        psi_3 = np.einsum('tij, kij -> tk', E_xxT[:-1], WWT)

        s = psi_1 + psi_2 + psi_3
        assert np.all(s >= 0)
        for itr in range(n_iter):
            lambda_bs = self.lambda_bs

            self.a = 2 * (m * lambda_bs).sum(axis=1) + K / 2.0 - 1.0
            self.a /= 2 * lambda_bs.sum(axis=1)

            self.bs = np.sqrt(s - 2 * m * self.a[:, None] + self.a[:, None] ** 2)

    def vb_E_step_discrete_states(self):
        self.clear_caches()

        trans_potential = np.exp(self.trans_distn.logpi)
        init_potential = self.pi_0
        likelihood_potential = self.vbem_aBl
        alphal = self._messages_forwards_log(trans_potential, init_potential, likelihood_potential)
        betal = self._messages_backwards_log(trans_potential, likelihood_potential)

        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:, None]
        np.exp(expected_states, out=expected_states)
        expected_states /= expected_states.sum(1)[:, None]

        Al = np.log(trans_potential)
        log_joints = alphal[:-1, :, None] + betal[1:, None, :] \
            + likelihood_potential[1:, None, :] + Al[None, :, :]
        log_joints -= log_joints.max(axis=(1, 2), keepdims=True)
        joints = np.exp(log_joints)
        joints /= joints.sum(axis=(1, 2), keepdims=True)

        normalizer = logsumexp(alphal[0] + betal[0])

        self.expected_states = expected_states
        self.expected_joints = joints
        self.expected_transcounts = joints.sum(0)
        self._normalizer = normalizer

        self.stateseq = self.expected_states.argmax(1).astype('int32')

        from pyslds.util import hmm_entropy
        params = (np.log(trans_potential), np.log(init_potential), likelihood_potential, normalizer)
        stats = (expected_states, self.expected_transcounts, normalizer)
        return hmm_entropy(params, stats)

    def expected_log_joint_probability(self):
        elp = np.dot(self.expected_states[0], np.log(self.pi_0))
        elp += np.sum(self.expected_joints * np.log(self.trans_matrix + 1e-16))

        elp += np.sum(self.expected_states * self.vbem_aBl)
        return elp

    def _init_vbem_from_gibbs(self):
        super(_SoftmaxRecurrentSLDSStatesBase, self)._init_mf_from_gibbs()
        self.vbem_update_auxiliary_vars()
        self.expected_joints = self.expected_states[:-1, :, None] * self.expected_states[1:, None, :]
        self._set_expected_trans_stats()


class SoftmaxRecurrentSLDSStates(_SoftmaxRecurrentSLDSStatesVBEM,
                                 _SoftmaxRecurrentSLDSStatesMeanField):
    pass

