from puppersim.reacher import reacher_env
import math
import time
import numpy as np
import copy

HIP_OFFSET = 0.0335
L1 = 0.08
L2 = 0.11


def calculate_forward_kinematics_robot(joint_angles):
  # compute end effector pos in cartesian cords given angles
  # joint_angles = np.reshape(joint_angles, (3,))

  x1 = L1 * math.sin(joint_angles[1])
  z1 = L1 * math.cos(joint_angles[1])

  x2 = L2 * math.sin(joint_angles[1] + joint_angles[2])
  z2 = L2 * math.cos(joint_angles[1] + joint_angles[2])

  foot_pos = np.array([[HIP_OFFSET], [x1 + x2], [z1 + z2]])

  rot_mat = np.array(
      [[math.cos(-joint_angles[0]), -math.sin(-joint_angles[0]), 0],
       [math.sin(-joint_angles[0]),
        math.cos(-joint_angles[0]), 0], [0, 0, 1]])

  end_effector_pos = np.matmul(rot_mat, foot_pos)

  xyz = np.transpose(end_effector_pos)

  return xyz[0]


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
