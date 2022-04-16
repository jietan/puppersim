from puppersim import reacher
import pytest
import numpy as np

from puppersim.reacher import reacher_kinematics


def test_IK_cost():
  target = np.array([0.0335, 0, 0.3])
  guess = np.array([0, 0, 0])
  cost = reacher_kinematics.ik_cost(target, guess)
  expected_cost = 0.5 * 0.11**2
  assert cost == pytest.approx(expected_cost, abs=1e-4)


def test_FK_IK_FK_consistency():
  """
  Tests that FK and IK are consistent. Does not test they are correct.
  
  Determines reachable point, x, by using FK on random joint angles. Then uses 
  IK to return a corresponding set of joint angles. Use FK to calculate the
  resulting point and compare against the original position, x.
  """
  joint_angles = np.random.uniform(-2*np.pi, 2*np.pi, 3)
  end_effector_pos_A = reacher_kinematics.calculate_forward_kinematics_robot(
      joint_angles)
  guess = np.array([0.0, 0.5, 0.5]) # Helpful to start far from singularity
  joint_angles_IK = reacher_kinematics.calculate_inverse_kinematics(
      end_effector_pos_A, guess)
  end_effector_pos_B = reacher_kinematics.calculate_forward_kinematics_robot(
      joint_angles_IK)
  print("Maximum absolute error: ",
        np.max(abs(end_effector_pos_A - end_effector_pos_B)))
  assert end_effector_pos_A == pytest.approx(end_effector_pos_B, abs=1e-4)
