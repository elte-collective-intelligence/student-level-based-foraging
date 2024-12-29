#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
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
#
# Authors: Ryan Shim, Gilbert

import math
import numpy
import csv


from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu



class DQNEnvironment(Node):
    def __init__(self, agent_id):
        super().__init__(f'dqn_environment_{agent_id}')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.agent_id = agent_id
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0

        self.action_size = 5
        self.done = False
        self.fail = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 1.0
        self.scan_ranges = []
        self.min_obstacle_distance = 10.0
        self.min_obstacle_angle = 10.0

        self.local_step = 0

        self.agent_id = agent_id
        self.collision_count = 0
        self.global_goal_counter = 0
        self.global_collision_counter = 0
        self.goal_count = 0
        self.episode_count = 0
        self.log_interval = 5
        self.cumulative_data = []

        

        # File for logging
        self.log_file = "/home/kwamboka/dqn_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/resource/goalcollisions.log"

         # Write CSV headers
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Goals', 'Collisions', 'Stage', 'Num_Agents'])

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Namespace for the agent
        namespace = f'/robot{self.agent_id}'

        # Initialise publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, f'{namespace}/cmd_vel', qos)

        # Initialise subscribers
        self.goal_pose_sub = self.create_subscription(
            Pose,
            f'{namespace}/goal_pose',
            self.goal_pose_callback,
            qos)
        self.odom_sub = self.create_subscription(
            Odometry,
            f'{namespace}/odom',
            self.odom_callback,
            qos)
        self.scan_sub = self.create_subscription(
            LaserScan,
            f'{namespace}/scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)

        # Initialise client
        self.task_succeed_client = self.create_client(
            Empty, f'{namespace}/task_succeed')
        self.task_fail_client = self.create_client(
            Empty, f'{namespace}/task_fail')

        self.dqn_com_server = self.create_service(
            Dqn, f'{namespace}/dqn_com', self.dqn_com_callback)


    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def goal_pose_callback(self, msg):
        self.goal_pose_x = msg.position.x
        self.goal_pose_y = msg.position.y

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x-self.last_pose_x)**2
            + (self.goal_pose_y-self.last_pose_y)**2)

        path_theta = math.atan2(
            self.goal_pose_y-self.last_pose_y,
            self.goal_pose_x-self.last_pose_x)

        goal_angle = path_theta - self.last_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.min_obstacle_distance = min(self.scan_ranges)
        self.min_obstacle_angle = numpy.argmin(self.scan_ranges)

    def get_state(self):
        state = list()
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        state.append(float(self.min_obstacle_distance))
        state.append(float(self.min_obstacle_angle))
        self.local_step += 1

        # Succeed
        if self.goal_distance < 0.20:  # unit: m
            print(f"Agent {self.agent_id}: Goal! :)")
            self.goal_count += 1
            self.global_goal_counter += 1
            self.succeed = True
            self.done = True
            self.log_data()  # Log the data
            self.cmd_vel_pub.publish(Twist())  # robot stop
            self.local_step = 0
            req = Empty.Request()
            # self.reset()
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_succeed_client.call_async(req)

        # Fail
        if self.min_obstacle_distance < 0.13:  # unit: m
            print(f"Agent {self.agent_id}: Collision! :(")
            self.collision_count += 1
            self.global_collision_counter += 1
            self.fail = True
            self.done = True
            self.log_data()
            self.cmd_vel_pub.publish(Twist())  # robot stop
            self.local_step = 0
            # self.reset()
            req = Empty.Request()
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_fail_client.call_async(req)

        if self.local_step == 500:
            print(f"Agent {self.agent_id}: Time out! :(")
            self.done = True
            self.local_step = 0
            req = Empty.Request()
            # self.reset()
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_fail_client.call_async(req)

        return state
    
    def log_data(self):
        self.episode_count += 1

        # Log data every `log_interval` episodes
        if self.episode_count % self.log_interval == 0:
            entry = {
                'Episode': self.episode_count,
                'Goals': self.global_goal_counter,
                'Collisions': self.global_collision_counter,
                'Stage': '1',
                'Num_Agents': 2  # Adjust if you add more agents
            }

            self.cumulative_data.append(entry)

            # Save to CSV
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    entry['Episode'],
                    entry['Goals'],
                    entry['Collisions'],
                    entry['Stage'],
                    entry['Num_Agents']
                ])

            print(f"Logged data: {entry}")

            # Reset counters
            self.goal_count = 0
            self.collision_count = 0

    def reset(self):
        """
        Resets each agent to its initial state.
        """
        # Reset agent-specific variables
        self.done = False
        self.succeed = False
        self.fail = False
        self.local_step = 0
        self.goal_count = 0
        self.collision_count = 0
        self.goal_pose_x = 0
        self.goal_pose_y = 0
        self.last_pose_x = 0
        self.last_pose_y = 0
        self.goal_distance = 1  # Reset goal distance (or set to initial value)
        self.goal_angle = 0  # Reset goal angle
        
        # Reset other agent-specific states (e.g., sensors, odometry, etc.)
        self.scan_ranges = []
        self.min_obstacle_distance = 10  # Set to a large value initially
        self.min_obstacle_angle = 0
        
        # Optionally reset the robot's position in the environment
    
    
        

    def imu_callback(self, msg):
        # Process the IMU message
        self.get_logger().info(f"Received IMU data")
    def dqn_com_callback(self, request, response):
        action = request.action
        twist = Twist()
        twist.linear.x = 0.3
        twist.angular.z = ((self.action_size - 1)/2 - action) * 1.5
        self.cmd_vel_pub.publish(twist)

        response.state = self.get_state()
        response.reward = self.get_reward(action)
        response.done = self.done

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        if request.init is True:
            self.init_goal_distance = math.sqrt(
                (self.goal_pose_x-self.last_pose_x)**2
                + (self.goal_pose_y-self.last_pose_y)**2)

        return response

    def get_reward(self, action):
        yaw_reward = 1 - 2*math.sqrt(math.fabs(self.goal_angle / math.pi))

        distance_reward = (2 * self.init_goal_distance) / \
            (self.init_goal_distance + self.goal_distance) - 1

        # Reward for avoiding obstacles
        if self.min_obstacle_distance < 0.25:
            obstacle_reward = -2
        else:
            obstacle_reward = 0

        reward = yaw_reward + distance_reward + obstacle_reward

        # + for succeed, - for fail
        if self.succeed:
            reward += 5
        elif self.fail:
            reward -= -10
        print(reward)

        return reward

    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)

    # Global counters for goals and collisions


    # Create a MultiThreadedExecutor
    executor = MultiThreadedExecutor()

    # Create multiple agents and add them to the executor
    agents = []
    for i in range(2):  # Modify the range to add more agents
        agent = DQNEnvironment(agent_id=i)
        agents.append(agent)
        executor.add_node(agent)

    try:
        # Spin the executor (handles multiple nodes concurrently)
        executor.spin()
    finally:
        # Shutdown nodes and executor
        for agent in agents:
            agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


