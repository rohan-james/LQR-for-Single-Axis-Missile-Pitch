# System Dynamics and Cost Functional Formulations

Consider a continuous-time linear time-invariant (LTI) dynamic system
governed by the state-space equation: $$\dot{x}(t) = A x(t) + B u(t)$$

To design an optimal control policy over a finite time horizon
$t \in [t_0, t_f]$, a quadratic performance index $J$ is defined in
scalar-matrix form:
$$J = \frac{1}{2} x(t_f)^T Q_f x(t_f) + \frac{1}{2} \int_{t_0}^{t_f} \left( x(t)^T Q(t) x(t) + 2 x(t)^T N(t) u(t) + u(t)^T R(t) u(t) \right) dt$$

For infinite-horizon regulator problems where $t_f \to \infty$, terminal
state constraints are relaxed, yielding the stationary performance
functional:
$$J = \frac{1}{2} \int_{0}^{\infty} \left( x(t)^T Q(t) x(t) + 2 x(t)^T N(t) u(t) + u(t)^T R(t) u(t) \right) dt$$

## Physical Interpretations of Cost Matrix Terms {#physical-interpretations-of-cost-matrix-terms .unnumbered}

-   $x(t_f)^T Q_f x(t_f)$ --- **Terminal Cost Weighting**: Penalizes
    residual state deviations at the final time $t_f$, ensuring precise
    trajectory completion.

-   $u(t)^T R(t) u(t)$ --- **Control Expenditure Penalty**: Penalizes
    control effort to prevent actuator saturation and reduce physical
    energy consumption.

-   $2 x(t)^T N(t) u(t)$ --- **Cross-Coupling Penalty**: Penalizes
    specific joint interactions between system states and control
    inputs.

## Solvability Constraints {#solvability-constraints .unnumbered}

To guarantee a well-posed optimization problem with a unique, stable
solution, the composite weighting matrix must be positive semi-definite,
and the control penalty matrix must be strictly positive definite:
$$\begin{bmatrix} Q & N \\ N^T & R \end{bmatrix} \ge 0 \quad \text{and} \quad R > 0$$

# Algebraic Riccati Equation and Numerical Solvers

For infinite-horizon Linear Quadratic Regulator (LQR) problems, the
optimal state-feedback gain matrix is derived from the symmetric
positive semi-definite solution matrix $P$ satisfying the Continuous
Algebraic Riccati Equation (CARE):
$$A^T P + P A + Q - P B R^{-1} B^T P = 0$$ Once the unique stabilizing
solution matrix $P$ is calculated, the optimal state-feedback control
law $u(t) = -K x(t)$ is established to stabilize the closed-loop system.

## Numerical Solution Methods for CARE

### Eigen-Decomposition Method (Direct Solvers)

This approach converts the non-linear quadratic matrix equation of
dimension $n \times n$ into a linear $2n \times 2n$ matrix problem
governed by the system Hamiltonian matrix $H$:
$$H = \begin{bmatrix} A & -B R^{-1} B^T \\ -Q & -A^T \end{bmatrix}$$

### Laub's Schur Method

Direct eigenvector computation can suffer from severe numerical
instability in the presence of repeated or ill-conditioned eigenvalues.
Laub's Schur method provides a numerically robust alternative by
employing an ordered real Schur decomposition of $H$:

1.  Compute the real Schur decomposition $H = U T U^T$, where $T$ is
    quasi-upper triangular and
    $U = \begin{bmatrix} U_{11} & U_{12} \\ U_{21} & U_{22} \end{bmatrix}$
    is orthogonal.

2.  Reorder $T$ using orthogonal Givens rotations such that the $n$
    stable eigenvalues satisfying $\text{Re}(\lambda) < 0$ occupy the
    upper-left $n \times n$ block of $T$.

3.  Extract the stable invariant subspace spanned by the first $n$
    columns of $U$ to compute $P$: $$P = U_{21} U_{11}^{-1}$$

# Missile Pitch Dynamics and Environmental Modeling

## Atmospheric and Physical Parameters

The physical geometry and atmospheric operating conditions governing
missile pitch dynamics are modeled as follows:

-   **Pitch Moment of Inertia**:
    $I_{yy} = \frac{1}{12} M L^2 + \frac{1}{4} M R^2$

-   **Ambient Temperature Profile**: $$T = \begin{cases} 
        216.65 \text{ K}, & \text{if } \text{altitude} > 11 \text{ km} \\
        288.15 - 0.0065 \times \text{altitude}, & \text{if } \text{altitude} \le 11 \text{ km}
        \end{cases}$$

-   **Speed of Sound**: $a = \sqrt{\gamma R T}$

-   **Flight Velocity**: $V = \text{Mach number} \times a$

-   **Atmospheric Pressure**:
    $p = 101325 \times e^{-h / H}, \quad H = 8500 \text{ m}$

-   **Air Density**: $\rho = 1.225 \times e^{-h / H}$

-   **Dynamic Pressure**: $\bar{q} = \frac{1}{2} \rho V^2$

## Dimensional Stability Derivatives

Normalizing dynamic parameters by missile mass $m$, reference area $S$,
mean aerodynamic chord $\bar{c}$, and velocity $V$ yields the
dimensional stability derivatives:
$$Z_\alpha = \frac{\bar{q} S}{m V} C_{Z\alpha}, \quad Z_\delta = \frac{\bar{q} S}{m V} C_{Z\delta}, \quad M_\alpha = \frac{\bar{q} S \bar{c}}{I_{yy}} C_{m\alpha}, \quad M_\delta = \frac{\bar{q} S \bar{c}}{I_{yy}} C_{m\delta}$$

## State-Space Representation

Choosing the state vector as $x(t) = [\Delta \alpha(t), \Delta q(t)]^T$
and control input as $u(t) = \Delta \delta(t)$, the linearized system
dynamics are formulated as:
$$A = \begin{bmatrix} Z_\alpha & 1 + Z_q \\ M_\alpha & M_q \end{bmatrix}, \quad B = \begin{bmatrix} Z_\delta \\ M_\delta \end{bmatrix}$$

# Linearization and Aerodynamic Derivatives

## Bryson's Rule for Cost Function Scaling

To account for unit discrepancies across state and control variables,
Bryson's rule scales each diagonal entry in $Q$ and $R$ by the inverse
square of its maximum allowable physical limit (with angular units
expressed in radians):
$$Q = \begin{bmatrix} \frac{1}{\alpha_{\max}^2} & 0 \\ 0 & \frac{1}{q_{\max}^2} \end{bmatrix}, \quad R = \begin{bmatrix} \frac{1}{\delta_{\max}^2} \end{bmatrix}$$

## Linearization Around a Trim Condition

Missile flight dynamics are non-linear. To design a linear quadratic
controller, the non-linear equations of motion are linearized about a
trimmed flight equilibrium condition $(x_0, u_0)$ using a first-order
Taylor series expansion:
$$\dot{x} \approx f(x_0, u_0) + \left.\frac{\partial f}{\partial x}\right|_0 \Delta x + \left.\frac{\partial f}{\partial u}\right|_0 \Delta u$$

**Short-Period Mode Analysis:** The pitch channel response is dominated
by the short-period mode, characterized by rapid rotational pitching and
fast variation in angle of attack $\alpha$. Matrix $A$ captures how the
system state rates $(\dot{\alpha}, \dot{q})$ evolve due to state
perturbations independent of control input.

### Pitch Acceleration ($\dot{q}$)

From Newton's second law for rotational dynamics about the missile pitch
axis ($Y_B$): $$I_{yy} \dot{q} = M_{\text{total}}(\alpha, q, \delta)$$
Linearizing about the trim point yields:
$$\Delta \dot{q} = \underbrace{\left( \frac{1}{I_{yy}} \frac{\partial M}{\partial \alpha} \right)}_{M_\alpha} \Delta \alpha + \underbrace{\left( \frac{1}{I_{yy}} \frac{\partial M}{\partial q} \right)}_{M_q} \Delta q$$

-   $M_\alpha \rightarrow$ **Pitch Stiffness / Static Longitudinal
    Stability**: Quantifies the restoring aerodynamic torque per unit
    change in $\alpha$.

    -   $M_\alpha < 0 \implies$ Statically stable (generates a
        stabilizing restoring moment toward zero angle of attack).

    -   $M_\alpha > 0 \implies$ Aerodynamically unstable (requires
        active feedback augmentation).

-   $M_q \rightarrow$ **Pitch Damping Derivative**: Quantifies
    aerodynamic resistance opposing pitch rate rotation.

### Angle-of-Attack Rate ($\dot{\alpha}$)

The rate of change of angle of attack is derived from fundamental
kinematic relations along the body vertical axis ($Z_B$). Kinematically:
$$\theta = \gamma + \alpha \implies \dot{\alpha} = \dot{\theta} - \dot{\gamma} = q - \dot{\gamma} \quad (\text{where } \gamma = \text{flight path angle})$$
Applying Newton's second law along the lift vector axis, the flight path
angle rate $\dot{\gamma}$ is driven by vertical aerodynamic force $F_z$:
$$m V \dot{\gamma} = -F_z \implies \dot{\gamma} = -\frac{F_z}{m V}$$

# Kinematic Relationships and Cost Functional Tuning

Combining the expressions for flight path rate and kinematics yields the
non-linear angle-of-attack derivative:
$$\dot{\alpha} = q + \frac{F_z(\alpha, q, \delta)}{m V}$$ Linearizing
the aerodynamic force contribution gives:
$$\frac{F_z}{m V} \approx \underbrace{\left( \frac{1}{m V} \frac{\partial F_z}{\partial \alpha} \right)}_{Z_\alpha} \Delta \alpha + \underbrace{\left( \frac{1}{m V} \frac{\partial F_z}{\partial q} \right)}_{Z_q} \Delta q$$
$$\implies \Delta \dot{\alpha} = Z_\alpha \Delta \alpha + (1 + Z_q) \Delta q$$

## Physical Interpretation of Terms {#physical-interpretation-of-terms .unnumbered}

-   **Kinematic Coupling ($1$):** Pure body rotation. In the absence of
    spatial velocity vector reorientation ($\dot{\gamma} = 0$), pitching
    at rate $q$ increases $\alpha$ at a $1:1$ ratio.

-   $Z_\alpha$ **(Force Derivative):** Vertical aerodynamic force
    generated per unit variation in $\alpha$. Under standard sign
    conventions ($Z$-axis pointing downward), lift generation makes
    $Z_\alpha$ strictly negative.

-   $Z_q$ **(Rotary Force Derivative):** Incremental normal force
    generated across aerodynamic surfaces during pitching rotation
    ($Z_q \ll 1$).

-   $M_\delta$ **(Control Effectiveness):** Control moment per unit fin
    deflection relative to inertia $I_{yy}$; serves as the primary
    control authority metric for pitch rotational acceleration.

-   $Z_\delta$ **(Direct Control Force Derivative):** Direct normal
    force generated immediately upon deflecting the control surface.

> **Note on Non-Minimum-Phase Behavior:** Deflecting a tail control fin
> to initiate an upward pitch initially exerts a downward normal force
> ($Z_\delta$). This creates a transient initial drop in trajectory
> prior to angle-of-attack buildup, producing non-minimum-phase dynamics
> (right-half-plane zeros).

## Cost Functional Tuning Strategies

The explicit quadratic cost functional is expressed as:
$$J = \int_0^\infty \left( x^T Q x + u^T R u \right) dt = \int_0^\infty \left( Q_{11} \Delta \alpha^2 + Q_{22} \Delta q^2 + R_{11} \Delta \delta^2 \right) dt$$

Selecting unweighted identity matrices ($Q = I$, $R = I$) introduces
severe scale distortion, as numerical values for pitch rates $\Delta q$
typically dwarf angle-of-attack perturbations $\Delta \alpha$. Applying
Bryson's normalization ($Q_{11} = \frac{1}{\Delta \alpha_{\max}^2}$)
scales non-dimensional errors into the unit range $[0, 1]$.

-   **Increasing $Q_{11}$**: Heavily penalizes angle-of-attack
    deviations, prioritizing strict trajectory tracking at the expense
    of aggressive control surface activity and higher actuator slew
    rates.

-   **Increasing $R_{11}$**: Heavily penalizes actuator movement,
    promoting smooth fin deflections and preventing actuator fatigue at
    the expense of slower transient tracking response.

# Aerodynamic Coefficient Definitions

-   $C_Z$ **(Vertical Force Coefficient):** Dimensionless aerodynamic
    force component acting along the vehicle's body $Z$-axis.

-   $C_m$ **(Pitching Moment Coefficient):** Dimensionless aerodynamic
    pitching torque about the center of gravity.

-   $C_{Z\alpha}$ **(Normal Force Curve Slope):** Sensitivity of
    vertical force coefficient with respect to angle of attack
    ($\alpha$).

-   $C_{Z\delta}$ **(Control Surface Force Coefficient):** Sensitivity
    of vertical force coefficient with respect to control surface
    deflection ($\delta$).

-   $C_{m\alpha}$ **(Static Longitudinal Stability Derivative):**
    Aerodynamic stiffness coefficient. A negative sign
    ($C_{m\alpha} < 0$) signifies inherent dynamic stability.

-   $C_{m\delta}$ **(Control Moment Effectiveness):** Sensitivity of
    pitching moment with respect to fin deflection ($\delta$), dictating
    dynamic maneuverability.

Mathematically, these non-dimensional aerodynamic coefficients are
calculated via partial differentiation:
$$C_{m\alpha} = \frac{\partial C_m}{\partial \alpha}$$
