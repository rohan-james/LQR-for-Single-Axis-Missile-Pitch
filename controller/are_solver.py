import math
import pytest
import numpy as np
import scipy

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class ARESolver:
    def __init__(self, state_dimension, control_dimension):
        self.n = state_dimension
        self.m = control_dimension

        self.H = None
        self.T = None
        self.U = None
        self.P = None
        self.K = None

    def construct_input_matrices(self, A=None, B=None, Q=None, R=None):
        if A is not None:
            self.A = np.asarray(A, dtype=float)
            self.B = np.asarray(B, dtype=float)
            self.Q = np.asarray(Q, dtype=float)
            self.R = np.asarray(R, dtype=float)

        else:
            self.A = np.random.randn(self.n, self.n)
            self.B = np.random.randn(self.n, self.m)

            raw_q = np.random.randn(self.n, self.n)
            self.Q = raw_q.T @ raw_q

            raw_r = np.random.randn(self.m, self.m)
            self.R = raw_r.T @ raw_r + np.eye(self.m) * 1e-2

    def construct_hamiltonian(self):
        R_inv_B_T = np.linalg.solve(self.R, self.B.T)
        top_right = self.B @ R_inv_B_T

        top_row = np.hstack([self.A, -top_right])
        bottom_row = np.hstack([-self.Q, -self.A.T])

        self.H = np.vstack([top_row, bottom_row])
        return self.H

    def compute_schur_form(self):
        if self.H is None:
            self.construct_hamiltonian()

        def select_stable(eignevalue):
            return eignevalue.real < 0.0

        ret = scipy.linalg.schur(self.H, sort=select_stable)
        self.T, self.U = ret[0], ret[1]
        return self.T, self.U

    def compute_matrix_P(self):
        if self.U is None:
            self.compute_schur_form()

        U11 = self.U[: self.n, : self.n]
        U21 = self.U[self.n :, : self.n]

        P_raw = np.linalg.solve(U11.T, U21.T).T
        self.P = 0.5 * (P_raw + P_raw.T)
        return self.P

    def return_optimal_gains(self):
        if self.P is None:
            self.compute_matrix_P()

        self.K = np.linalg.solve(self.R, self.B.T @ self.P)
        return self.K

    def compute_residual(self):
        if self.P is None:
            self.compute_matrix_P()

        R_inv_BT = np.linalg.solve(self.R, self.B.T)
        residual = (
            self.A.T @ self.P
            + self.P @ self.A
            - self.P @ self.B @ R_inv_BT @ self.P
            + self.Q
        )
        return np.linalg.norm(residual, ord="fro")

    def verify_sol(self):
        P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.solve(self.R, self.B.T @ P)

        return K
