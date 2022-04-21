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
  """Compute end effector pos in cartesian cords given angles

  Args:
    joint_angles: np array with elements (base, shoulder, elbow). Radians
  
  Returns:
    Position of end-effector in meters, np array (x, y, z)
  """
  base_angle = -joint_angles[0]
  shoulder_angle = -joint_angles[1]
  elbow_angle = -joint_angles[2]

  y1 = L1 * math.sin(shoulder_angle)
  z1 = L1 * math.cos(shoulder_angle)

  y2 = L2 * math.sin(shoulder_angle + elbow_angle)
  z2 = L2 * math.cos(shoulder_angle + elbow_angle)

  foot_pos = np.array([[HIP_OFFSET], [y1 + y2], [z1 + z2]])

  rot_mat = np.array([[math.cos(base_angle), -math.sin(base_angle), 0],
                      [math.sin(base_angle),
                       math.cos(base_angle), 0], [0, 0, 1]])

  end_effector_pos = rot_mat @ foot_pos
  return end_effector_pos[:, 0]


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
