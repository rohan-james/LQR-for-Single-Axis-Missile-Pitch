import numpy as np
import scipy
import pytest
import itertools

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from controller.are_solver import ARESolver
from model.single_axis_pitch_model import SingleAxisMissilePitch


def test_initialization_defaults():
    missile = SingleAxisMissilePitch()
    assert missile.density > 0.0
    assert missile.q_bar > 0.0
    assert missile.velocity > 0.0


def test_stratosphere_temperature_boundary():
    missile_low = SingleAxisMissilePitch(altitude=5000.0)
    missile_high = SingleAxisMissilePitch(altitude=15000.0)

    assert missile_low.temperature == pytest.approx(255.65, abs=1e-2)
    assert missile_high.temperature == pytest.approx(216.65, abs=1e-2)


def test_matrix_generation_and_dimensions():
    missile = SingleAxisMissilePitch()
    missile.compute_state_and_control_matrices()

    assert missile.A.shape == (2, 2)
    assert missile.B.shape == (2, 1)
    assert missile.A[0, 0] < 0.0


# ARE Residual & SciPy Cross-Verification Grid Sweep
@pytest.mark.parametrize(
    "altitude, mach, mass",
    list(
        itertools.product(
            [1000.0, 5000.0, 10000.0, 15000.0, 20000.0],
            [1.5, 2.5, 3.5, 4.5],
            [110.0, 180.0, 250.0],
        )
    ),
)
def test_closed_loop_stability_across_envelope(altitude, mach, mass):

    missile = SingleAxisMissilePitch(altitude=altitude, mach_number=mach, mass=mass)
    missile.compute_state_and_control_matrices()
    missile.setup_cost_weight_matrix()

    ctrb_matrix = np.hstack([missile.B, missile.A @ missile.B])
    assert np.linalg.matrix_rank(ctrb_matrix) == 2

    solver = ARESolver(state_dimension=2, control_dimension=1)
    solver.construct_input_matrices(missile.A, missile.B, missile.Q, missile.R)

    P_custom = solver.compute_matrix_P()
    K_custom = solver.return_optimal_gains()
    K_scipy = solver.verify_sol()
    residual = solver.compute_residual()

    assert residual < 1e-6, (
        f"ARE residual norm exceeded threshold ({residual:.2e}) at "
        f"Alt={altitude}m, Mach={mach}, Mass={mass}kg"
    )

    np.testing.assert_allclose(
        K_custom,
        K_scipy,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"Custom ARE solver mismatch vs SciPy at Alt={altitude}m, Mach={mach}, Mass={mass}kg",
    )


# Robustness & Parameter-Sensitivity Test
def test_parameter_sensititvity_and_margins():
    nominal = SingleAxisMissilePitch(altitude=10000.0, mach_number=2.5, mass=180.0)
    nominal.compute_state_and_control_matrices()
    nominal.setup_cost_weight_matrix()

    solver = ARESolver(state_dimension=2, control_dimension=1)
    solver.construct_input_matrices(nominal.A, nominal.B, nominal.Q, nominal.R)
    K_nominal = solver.return_optimal_gains()

    perturbations = [-0.25, 0.0, 0.25]
    for d_z, d_ma, d_md in itertools.product(
        perturbations, perturbations, perturbations
    ):
        p_missile = SingleAxisMissilePitch(
            altitude=10000.0,
            mach_number=2.5,
            mass=180.0,
            normal_force_alpha_derivative=-12.0 * (1.0 + d_z),
            pitching_moment_alpha_derivative=-15.0 * (1.0 + d_z),
            pitch_control_authority=-20.0 * (1.0 + d_z),
        )
        p_missile.compute_state_and_control_matrices()

        A_cl_perturbed = p_missile.A - p_missile.B @ K_nominal
        poles = np.linalg.eigvals(A_cl_perturbed)

        assert np.all(np.real(poles) < 0.0), (
            f"System went unstable under aero perturbations: "
            f"dCZ_a={d_z:.0%}, dCm_a={d_ma:.0%}, dCm_d={d_md:.0%}. Poles: {poles}"
        )

    for gain_factor in [0.5, 2.0]:
        A_cl_gain_varied = nominal.A - (gain_factor * nominal.B) @ K_nominal
        poles = np.linalg.eigvals(A_cl_gain_varied)
        assert np.all(
            np.real(poles) < 0.0
        ), f"Unstable at gain multiplier {gain_factor}"


# Flight Envelope Failure Handling Sweep
def test_flight_envelope_failure_handling():
    altitudes = [1000.0, 5000.0, 10000.0, 15000.0, 20000.0]
    machs = [1.5, 2.5, 3.5, 4.5]
    masses = [110.0, 180.0, 250.0]

    failures = []

    for alt, mach, mass in itertools.product(altitudes, machs, masses):
        try:
            missile = SingleAxisMissilePitch(altitude=alt, mach_number=mach, mass=mass)
            missile.compute_state_and_control_matrices()
            missile.setup_cost_weight_matrix()

            if np.isnan(missile.A).any() or np.isnan(missile.B).any():
                failures.append((alt, mach, mass, "NaN detected in system matrices"))
                continue

            ctrb = np.hstack([missile.B, missile.A @ missile.B])
            if np.linalg.matrix_rank(ctrb) < 2:
                failures.append((alt, mach, mass, "System lost controllability"))
                continue

            solver = ARESolver(2, 1)
            solver.construct_input_matrices(missile.A, missile.B, missile.Q, missile.R)
            K = solver.return_optimal_gains()

            if np.isnan(K).any():
                failures.append((alt, mach, mass, "Gain matrix K contains NaN"))

        except Exception as e:
            failures.append(
                (alt, mach, mass, f"Raised exception: {type(e).__name__} - {str(e)}")
            )

    if failures:
        msg = "\n".join(
            [
                f"  Alt={f[0]}m, Mach={f[1]}, Mass={f[2]}kg -> Reason: {f[3]}"
                for f in failures
            ]
        )
        pytest.fail(
            f"Flight envelope sweep failed at {len(failures)} configurations:\n{msg}"
        )


# Golden-Value Regression Test
def test_golden_value_regression():
    missile = SingleAxisMissilePitch(
        altitude=10000.0,
        reference_area=0.04,
        chord_length=0.225,
        mass=180.0,
        radius=0.113,
        length=3.65,
        mach_number=2.5,
        normal_force_alpha_derivative=-12.0,
        pitching_moment_alpha_derivative=-15.0,
        pitch_control_authority=-20.0,
        pitch_control_force_derivative=-2.0,
    )
    missile.compute_state_and_control_matrices()
    missile.setup_cost_weight_matrix()

    expected_A = np.array([[-0.37706949, 0.99000000], [-71.30889763, -2.00000000]])

    expected_B = np.array([[-0.06284491], [-95.07853018]])

    expected_Q = np.array([[32.82806350, 0.0], [0.0, 1.31312254]])

    expected_R = np.array([[8.20701588]])

    np.testing.assert_allclose(missile.A, expected_A, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(missile.B, expected_B, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(missile.Q, expected_Q, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(missile.R, expected_R, rtol=1e-5, atol=1e-5)

    solver = ARESolver(2, 1)
    solver.construct_input_matrices(missile.A, missile.B, missile.Q, missile.R)
    K = solver.return_optimal_gains()

    expected_K = np.array([[1.9701726, 0.2872338]])
    np.testing.assert_allclose(K, expected_K, rtol=1e-4, atol=1e-4)


# Stability Margins
def test_lqr_theoretical_stability_margins():
    missile = SingleAxisMissilePitch(altitude=10000.0, mach_number=2.5, mass=180.0)
    missile.compute_state_and_control_matrices()
    missile.setup_cost_weight_matrix()

    solver = ARESolver(2, 1)
    solver.construct_input_matrices(missile.A, missile.B, missile.Q, missile.R)
    K = solver.return_optimal_gains()

    gm_db, pm_deg, w_gc = missile.compute_stability_margins(K)

    assert (
        pm_deg >= 60.0
    ), f"Phase margin failed LQR guarantee: {pm_deg:.2f} deg < 60 deg"

    assert gm_db > 6.0, f"Gain margin below expected 6 dB threshold: {gm_db:.2f} dB"
