import numpy as np
import scipy.integrate

from controller.are_solver import ARESolver

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class SingleAxisMissilePitch:
    def __init__(
        self,
        altitude=10000.0,
        reference_area=0.04,
        chord_length=0.225,
        mass=180.0,
        radius=0.113,
        length=2.65,
        mach_number=2.5,
        normal_force_alpha_derivative=-12.0,
        pitching_moment_alpha_derivative=-15.0,
        pitch_control_authority=-20.0,
        pitch_control_force_derivative=-2.0,
    ):
        self.mach_number = mach_number
        self.altitude = altitude
        self.S = reference_area
        self.c_bar = chord_length
        self.m = mass
        self.radius = radius
        self.length = length
        self.c_z_alpha = normal_force_alpha_derivative
        self.c_m_alpha = pitching_moment_alpha_derivative
        self.c_m_delta = pitch_control_authority
        self.c_z_delta = pitch_control_force_derivative
        if self.altitude <= 11000.0:
            self.temperature = 288.15 - 0.0065 * self.altitude
        else:
            self.temperature = 216.65
        self.speed_of_sound = np.sqrt(1.4 * 287.05 * self.temperature)
        self.velocity = self.mach_number * self.speed_of_sound
        self.pressure = 101325.0 * np.exp(-self.altitude / 8500)
        self.density = 1.225 * np.exp(-self.altitude / 8500)
        self.q_bar = 0.5 * (self.density * self.velocity**2)
        self.pitch_inertia = (1.0 / 12.0) * self.m * (self.length**2) + (
            1.0 / 4.0
        ) * self.m * (self.radius**2)

        self.A = None
        self.B = None
        self.Q = None
        self.R = None

    def compute_state_and_control_matrices(self):

        Z_q = -0.01
        M_q = -2.0

        Z_alpha = ((self.q_bar * self.S) / (self.m * self.velocity)) * self.c_z_alpha
        Z_delta = ((self.q_bar * self.S) / (self.m * self.velocity)) * self.c_z_delta
        M_alpha = (
            (self.q_bar * self.S * self.c_bar) / self.pitch_inertia
        ) * self.c_m_alpha
        M_delta = (
            (self.q_bar * self.S * self.c_bar) / self.pitch_inertia
        ) * self.c_m_delta

        self.A = np.array([[Z_alpha, 1.0 + Z_q], [M_alpha, M_q]])
        self.B = np.array([[Z_delta], [M_delta]])

    def setup_cost_weight_matrix(self):
        max_alpha_deg = 10.0
        max_q_deg_s = 50.0
        max_fin_deg = 20.0

        max_alpha_rad = np.radians(max_alpha_deg)
        max_q_rad_s = np.radians(max_q_deg_s)
        max_fin_rad = np.radians(max_fin_deg)

        self.Q = np.diag(
            [
                1.0 / (max_alpha_rad**2),
                1.0 / (max_q_rad_s**2),
            ]
        )

        self.R = np.array([[1.0 / (max_fin_rad**2)]])

    def compute_stability_margins(self, K, omega_vec=np.logspace(-1, 4, 10000)):
        I = np.eye(self.A.shape[0])
        L_jw = np.zeros(len(omega_vec), dtype=complex)

        for i, w in enumerate(omega_vec):
            s = 1j * w
            L_jw[i] = (K @ np.linalg.solve(s * I - self.A, self.B))[0, 0]

        mag = np.abs(L_jw)
        phase = np.unwrap(np.angle(L_jw)) * (180.0 / np.pi)

        idx_gc = np.argmin(np.abs(mag - 1.0))
        w_gc = omega_vec[idx_gc]
        phase_margin_deg = 180.0 + phase[idx_gc]

        idx_pc = np.argmin(np.abs(phase - (-180.0)))
        if np.abs(phase[idx_pc] - (-180.0)) < 2.0:
            gain_margin_db = -20.0 * np.log10(mag[idx_pc])
        else:
            gain_margin_db = np.inf

        return gain_margin_db, phase_margin_deg, w_gc

    def run_simulation(self):
        areSolver = ARESolver(state_dimension=2, control_dimension=1)
        self.compute_state_and_control_matrices()
        self.setup_cost_weight_matrix()
        areSolver.construct_input_matrices(self.A, self.B, self.Q, self.R)
        P = areSolver.compute_matrix_P()
        K = areSolver.return_optimal_gains()
        res_norm = areSolver.compute_residual()
        K_verify = areSolver.verify_sol()

        A_cl = self.A - self.B @ K
        cl_poles = np.linalg.eigvals(A_cl)

        print(f"Computed Gain Matrix K:          {K.flatten()}")
        print(f"SciPy Verification Gain Matrix: {K_verify.flatten()}")
        print(f"CARE Residual Frobenius Norm:   {res_norm:.2e}")
        print(f"Closed-Loop Poles:              {cl_poles[0]:.2f}, {cl_poles[1]:.2f}")

        def closed_loop_dynamics(t, x):
            u = -K @ x
            return (self.A @ x + self.B @ u).flatten()

        x0 = np.array([np.radians(5.0), 0.0])
        t_span = (0.0, 2.0)
        t_eval = np.linspace(t_span[0], t_span[1], 500)

        sol = scipy.integrate.solve_ivp(
            closed_loop_dynamics, t_span, x0, t_eval=t_eval, method="RK45"
        )

        u_history = np.array([-K @ x for x in sol.y.T]).flatten()
