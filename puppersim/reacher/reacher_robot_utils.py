from serial.tools import list_ports
import time
import numpy as np


def get_serial_port():
  for device in list_ports.grep(".*"):
    if device.manufacturer == "Teensyduino":
      return device.device


def blocking_move(hardware_interface,
                  goal,
                  traverse_time,
                  dt=0.02,
                  kp=4.0,
                  kd=1.0,
                  max_current=3.0,
                  leg_index=3):
  """
  Defaults to controller back left leg
  """
  last_command = time.time()
  time_start = time.time()
  hardware_interface.set_joint_space_parameters(kp=kp,
                                                kd=kd,
                                                max_current=max_current)
  hardware_interface.read_incoming_data()
  initial_position = np.array(hardware_interface.robot_state.position[6:9])
  while (1):
    time.sleep(dt)
    now = time.time()
    last_command = now

    progress = (now - time_start) / traverse_time
    print(f"Move progress: {progress:0.2f}")
    next_position = progress * goal + (1 - progress) * initial_position

    full_actions = np.zeros([3, 4])
    full_actions[:, leg_index] = np.reshape(next_position, 3)
    hardware_interface.set_actuator_postions(np.array(full_actions))

    if now - time_start > traverse_time:
      return
