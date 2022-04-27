import math
import time
import numpy as np

HIP_OFFSET = 0.0335
L1 = 0.08
L2 = 0.11


def random_reachable_points(N, base_lb=-np.pi/4, base_ub=np.pi/4, shoulder_lb=0, shoulder_ub=np.pi/2,
                            elbow_lb=0, elbow_ub=np.pi/2):
  """Bounds are in radians"""
  X = []
  for i in range(N):
    target_angles = np.concatenate([
        np.random.uniform(base_lb, base_ub, 1),
        np.random.uniform(shoulder_lb, shoulder_ub, 1),
        np.random.uniform(elbow_lb, elbow_ub, 1)
    ])
    target = calculate_forward_kinematics_robot(target_angles)
    X.append(target)
  return X


def calculate_forward_kinematics_robot(joint_angles):
    """Calculate xyz coordinates of end-effector given joint angles.
    Use forward kinematics equations to calculate the xyz coordinates of the end-effector
    given some joint angles.
    Args:
      joint_angles: numpy array of 3 elements [base, shoulder, elbow]. Numpy array of 3 elements.
    Returns:
      xyz coordinates (in meters) of the end-effector in the arm frame. Numpy array of 3 elements.
    """
    theta_1, theta_2, theta_3 = joint_angles
    r_c0_e = np.array([0, 0, L2])
    R_b_c = np.array(
      [
        [math.cos(theta_3), 0, -math.sin(theta_3)],
        [0, 1, 0],
        [math.sin(theta_3), 0, math.cos(theta_3)]
      ]
    )
    R_a_b = np.array(
      [
        [math.cos(theta_2), 0, -math.sin(theta_2)],
        [0, 1, 0],
        [math.sin(theta_2), 0, math.cos(theta_2)]
      ]
    )
    R_n_a = np.array(
      [
        [math.cos(-theta_1), -math.sin(-theta_1), 0],
        [math.sin(-theta_1), math.cos(-theta_1), 0],
        [0, 0, 1]
      ]
    )
    r_b0_e = np.array([0, 0, L1]) + np.dot(R_b_c, r_c0_e)
    r_a0_e = np.array([0, -HIP_OFFSET, 0]) + np.dot(R_a_b, r_b0_e)
    return np.dot(R_n_a, r_a0_e)


def ik_cost(end_effector_pos, guess):
  return 0.5 * np.linalg.norm(
      calculate_forward_kinematics_robot(guess) - end_effector_pos)**2


def calculate_jacobian(joint_angles, delta):
  J = np.zeros((3, 3))
  for j in range(3):
    perturbation = np.zeros(3)
    perturbation[j] = delta
    J[:, j] = (calculate_forward_kinematics_robot(joint_angles + perturbation) -
               calculate_forward_kinematics_robot(joint_angles)) / delta
  return J


# TODO: Fix bugs and put into inverse kinematics
# def line_search(fun,
#                 initial_point,
#                 step_direction,
#                 gradient,
#                 alpha=0.1,
#                 beta=0.8):
#   t = 1
#   while fun(initial_point + t * step_direction
#            ) > fun(initial_point) + alpha * t * gradient.T @ step_direction:
#     t = t * beta

#   return initial_point + t * step_direction


def calculate_inverse_kinematics(end_effector_pos,
                                 guess,
                                 delta=1e-6,
                                 max_iters=30,
                                 eps=1e-8,
                                 verbose=False):
  """
  Compute joint angles given a desired end effector position.

  Uses newton method.

  Args:
    end_effector_pos: target end effector position
    guess: initial guess at joint angles
    delta: perturbation size for finite diffing
    max_iters: maximum iterations of gradient descent to run before stopping
    eps: stops gradient descent if cost decreases by less than this amount after a step
  Returns:
    Joint angles
  """
  if verbose:
    now = time.time()
  previous_cost = np.inf
  cost = 0.0
  for iters in range(max_iters):
    iters += 1
    J = calculate_jacobian(guess, delta=delta)
    residual = calculate_forward_kinematics_robot(guess) - end_effector_pos
    guess = guess - np.linalg.pinv(J) @ residual  # Take full Newton step
    cost = ik_cost(end_effector_pos, guess)
    if abs(previous_cost - cost) < eps:
      break
    previous_cost = cost
    if verbose:
      print("iters", iters, "cost: ", cost, "time elapsed:", time.time() - now)
  return guess
