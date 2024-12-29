#!/usr/bin/env python3

#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates


class Moving(Node):
    def __init__(self):
        super().__init__('moving_obstacle')  # Initialize the ROS2 node
        self.publisher_ = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.timer = self.create_timer(0.1, self.moving)  # Run the moving logic at a regular interval
        self.model_states_sub = self.create_subscription(
            ModelStates, 'gazebo/model_states', self.model_states_callback, 10
        )
        self.current_model_states = None
        self.get_logger().info("Moving obstacle node has been started.")

    def model_states_callback(self, msg):
        """Callback to update current model states."""
        self.current_model_states = msg

    def moving(self):
        """Publish new model state to move the obstacle."""
        if self.current_model_states is None:
            return

        try:
            index = self.current_model_states.name.index('obstacle')
            obstacle_state = ModelState()
            obstacle_state.model_name = 'obstacle'
            obstacle_state.pose = self.current_model_states.pose[index]
            obstacle_state.twist = Twist()
            obstacle_state.twist.angular.z = 0.5  # Rotate the obstacle
            self.publisher_.publish(obstacle_state)
        except ValueError:
            self.get_logger().warn("Obstacle model not found in gazebo/model_states.")
        except Exception as e:
            self.get_logger().error(f"Error while moving obstacle: {e}")


def main(args=None):
    rclpy.init(args=args)
    moving_obstacle_node = Moving()
    try:
        rclpy.spin(moving_obstacle_node)
    except KeyboardInterrupt:
        pass
    finally:
        moving_obstacle_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
