# Lint as: python3
"""A scene containing only a planar floor."""

from typing import Sequence
import random

import gin
from pybullet_envs.minitaur.envs_v2 import base_client
from pybullet_envs.minitaur.envs_v2.scenes import scene_base

_PLANE_URDF = ("plane.urdf")


@gin.configurable
class BumpyScene(scene_base.SceneBase):
  """A scene containing only a planar floor."""

  def __init__(self,
               *args,
               height_perturbation_range=0.03,
               lateral_friction=1.0,
               **kwargs):
    self._height_perturbation_range = height_perturbation_range
    self._lateral_friction = lateral_friction
    super().__init__(*args, **kwargs)

  def build_scene(self, pybullet_client):
    super().build_scene(pybullet_client)
    """
    visual_shape_id = self._pybullet_client.createVisualShape(
        shapeType=self._pybullet_client.GEOM_PLANE)
    collision_shape_id = self._pybullet_client.createCollisionShape(
        shapeType=self._pybullet_client.GEOM_PLANE)
    ground_id = self._pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id)
    self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=1.0)
    self.add_object(ground_id, scene_base.ObjectType.GROUND)
    """

    heightPerturbationRange = self._height_perturbation_range
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
    for j in range(int(numHeightfieldColumns / 2)):
      for i in range(int(numHeightfieldRows / 2)):
        height = random.uniform(0, heightPerturbationRange)
        heightfieldData[2 * i + 2 * j * numHeightfieldRows] = height
        heightfieldData[2 * i + 1 + 2 * j * numHeightfieldRows] = height
        heightfieldData[2 * i + (2 * j + 1) * numHeightfieldRows] = height
        heightfieldData[2 * i + 1 + (2 * j + 1) * numHeightfieldRows] = height

    terrainShape = self._pybullet_client.createCollisionShape(
        shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
        meshScale=[.05, .05, 1],
        heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
        heightfieldData=heightfieldData,
        numHeightfieldRows=numHeightfieldRows,
        numHeightfieldColumns=numHeightfieldColumns)
    ground_id = self._pybullet_client.createMultiBody(0, terrainShape)
    self.add_object(ground_id, scene_base.ObjectType.GROUND)

    self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=self._lateral_friction)

  @property
  def vectorized_map(self) -> Sequence[scene_base.Polygon]:
    """Returns vectorized map containing a list of polygon obstacles."""
    return []