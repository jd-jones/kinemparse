import numpy as np

from blocks.core import geometry


def updateCovariance(R_new, P, gyro_cov, sample_period, sqrt_mode=False):
    """ NOTE: if sqrt_mode, then P and gyro_cov are assumed to be the
    square roots of these respective matrices.
    """

    G = sample_period * R_new
    if sqrt_mode:
        # TODO
        pass
    else:
        P_new = P + G @ gyro_cov @ G.T

    return P_new


def updateOrientation(omega, R, sample_period):
    R_new = geometry.exponentialMap(sample_period * omega) @ R
    return R_new


def timeUpdate(omega, gyro_cov, R, P, sample_period, sqrt_mode=False):
    R_new = updateOrientation(omega, R, sample_period)
    P_new = updateCovariance(
        R_new, P, gyro_cov, sample_period,
        sqrt_mode=sqrt_mode
    )

    return R_new, P_new


def updatePosition(a, a_prev, a_cov, v, x, R, P, T, stationary_thresh=0.005):
    """ State-space update to velocity and position estimates.
    a: accel, t
    v: velocity, t - 1
    x: position, t - 1
    R: orientation, t
    T: sample period
    """

    delta_a = a - a_prev
    is_stationary = np.linalg.norm(delta_a) < stationary_thresh

    if is_stationary:
        v_new = np.zeros(3)
        x_new = x
    else:
        a_compensated = R @ a - gravityVec()
        v_new = v + T * a_compensated
        x_new = x + T * v + 0.5 * T ** 2 * a_compensated

    return v_new, x_new


def gravityVec():
    g = np.zeros(3)
    g[2] = 1

    return g


def measurementUpdate(a, accel_cov, R, P, sqrt_mode=False):
    g = gravityVec()
    a_est = - R @ g
    G = geometry.skewSymmetricMatrix(g)
    H = - R @ G
    # NOTE: H is always rank-deficient because the gravity vector only has one
    #   nonzero entry. This means the skew-symmetric matrix G will have one
    #   row and one column which are all zero.

    # FIXME: Construct S, S_inv from matrix square root of P
    if sqrt_mode:
        pass

    S = H @ P @ H.T
    # pinv is a hack. S is singular because of the issue with H above.
    S_inv = np.linalg.pinv(S)
    K = P @ H.T @ S_inv

    deviation_angle = K @ (a - a_est)
    R_new = geometry.exponentialMap(deviation_angle) @ R
    P_new = P - K @ S @ K.T

    return R_new, P_new, deviation_angle


def matrixSquareRoot(psd_matrix):
    # FIXME: This doesn't need to exist. Just compute the Cholesky factorization.
    w, v = np.linalg.eigh(psd_matrix)
    w_sqrt = np.sqrt(w)

    # A = X @ X.T
    #   = V @ W @ V.T
    # Therefore,
    # X = V @ sqrt(W)
    return v @ np.diag(w_sqrt)


def estimateOrientation(
        angular_velocities, linear_accels=None,
        gyro_cov=None, accel_cov=None,
        init_orientation=None, init_cov=None,
        init_velocity=None, init_position=None,
        sample_period=0.02, sqrt_mode=False):
    """ Estimate the orientation using a linear approximation (EKF). """

    if init_orientation is None:
        init_angle = np.zeros(3)
        init_orientation = np.eye(3)

    if init_cov is None:
        init_cov = np.eye(3)

    if gyro_cov is None:
        gyro_cov = np.eye(3)

    if accel_cov is None:
        accel_cov = np.eye(3)

    if init_velocity is None:
        init_velocity = np.zeros(3)

    if init_position is None:
        init_position = np.zeros(3)

    orientations = []  # [init_orientation.copy()]
    covariances = []  # [gyro_cov.copy()]
    angles = []  # [init_angle.copy()]

    velocities = []
    positions = []

    if sqrt_mode:
        gyro_cov = matrixSquareRoot(gyro_cov)
        accel_cov = matrixSquareRoot(accel_cov)

    R = init_orientation.copy()
    P = init_cov.copy()
    angle = init_angle.copy()

    v = init_velocity.copy()
    x = init_position.copy()

    # omega_prev = np.zeros(3)
    a_prev = np.zeros(3)

    for omega, a in zip(angular_velocities, linear_accels):
        R_new, P_new = timeUpdate(omega, gyro_cov, R, P, sample_period, sqrt_mode=sqrt_mode)
        angle += omega * sample_period

        v_new, x_new = updatePosition(a, a_prev, accel_cov, v, x, R_new, P_new, sample_period)

        if linear_accels is not None:
            R_new, P_new, deviation_angle = measurementUpdate(
                a, accel_cov, R_new, P_new,
                sqrt_mode=sqrt_mode
            )
            angle += deviation_angle

        R = R_new.copy()
        P = P_new.copy()

        v = v_new.copy()
        x = x_new.copy()

        # omega_prev = omega
        a_prev = a

        if sqrt_mode:
            P_new = P_new @ P_new.T

        orientations.append(R_new)
        covariances.append(P_new)
        angles.append(angle.copy())

        velocities.append(v_new)
        positions.append(x_new)

    angles = np.row_stack(tuple(angles))
    velocities = np.row_stack(tuple(velocities))
    positions = np.row_stack(tuple(positions))

    return orientations, covariances, angles, velocities, positions


def isStationary(gyro_seq, thresh=1.5):
    gyro_mag = np.linalg.norm(gyro_seq, axis=1)
    return gyro_mag < thresh


def subtractStationaryMean(sample_seq):
    is_stationary = isStationary(sample_seq)
    stationary_samples = sample_seq[is_stationary, :]
    stationary_mean = stationary_samples.mean(axis=0)

    return sample_seq - stationary_mean
