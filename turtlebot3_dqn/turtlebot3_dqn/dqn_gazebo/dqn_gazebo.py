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

import os
import random
import sys

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty


class DQNGazebo(Node):
    def __init__(self, stage, num_agents=2):
        super().__init__('dqn_gazebo')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.stage = int(stage)
        self.num_agents = num_agents

        def get_package_share_directory(package_name):
            from ament_index_python.packages import get_package_share_directory as get_pkg_share_dir
            return get_pkg_share_dir(package_name)

        # Path to goal model
        self.entity_dir_path = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'models/turtlebot3_dqn_world/goal_box'
        )
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        self.entity = open(self.entity_path, 'r').read()

        # Store goal positions and entity names for each agent
        self.goal_poses = [{"x": 0.5, "y": 0.0} for _ in range(num_agents)]
        self.entity_names = [f"goal_{i}" for i in range(num_agents)]

        self.init_state = False

        """************************************************************
        ** Initialise ROS publishers, subscribers, and clients
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Publishers
        self.goal_pose_pub = [
            self.create_publisher(Pose, f'robot{i}/goal_pose', qos) for i in range(num_agents)
        ]

        # Clients
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        # Namespace for the agent
        # namespace = f'/robot{self.agent_id}'

        # Servers
        # Empty,
        #     f'robot1/task_succeed',
        #     self.task_succeed_callback)
        for i in range(num_agents):
            # Create task_succeed service dynamically for each robot
            self.task_succeed_server = self.create_service(
                Empty,
                f'robot{i}/task_succeed',
                self.task_succeed_callback
            )
            
            # Create task_fail service dynamically for each robot
            self.task_fail_server = self.create_service(
                Empty,
                f'robot{i}/task_fail',
                self.task_fail_callback
    )



        # self.task_succeed_server = self.create_service(
        #     Empty,
        #     f'robot0/task_succeed',
        #     self.task_succeed_callback)
        # self.task_fail_server = self.create_service(Empty, f'robot0/task_fail', self.task_fail_callback)
        # self.task_succeed_server = self.create_service(
        #     Empty,
        #     f'robot1/task_succeed',
        #     self.task_succeed_callback)
        # self.task_fail_server = self.create_service(Empty, f'robot1/task_fail', self.task_fail_callback)

        # Timer
        self.publish_timer = self.create_timer(0.010, self.publish_callback)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def publish_callback(self):
        # Initialize state if not done already
        if not self.init_state:
            self.delete_entities()
            self.reset_simulation()
            self.init_state = True
            print("Initialized environment with multiple agents and goals.")

        # Publish goal poses and spawn entities
        for i in range(self.num_agents):
            goal_pose = Pose()
            goal_pose.position.x = self.goal_poses[i]["x"]
            goal_pose.position.y = self.goal_poses[i]["y"]
            self.goal_pose_pub[i].publish(goal_pose)
            print(f"Publishing goal for agent {i}: {goal_pose.position.x}, {goal_pose.position.y}")
            self.spawn_entity(self.entity_names[i], goal_pose)

    def task_succeed_callback(self, request, response):
        self.delete_entities()
        self.generate_goal_poses()
        print("Generated new goals for all agents.")

        return response

    def task_fail_callback(self, request, response):
        self.delete_entities()
        self.reset_simulation()
        self.generate_goal_poses()
        print("Reset the gazebo environment and generated new goals.")

        return response

    def generate_goal_poses(self):
        # Generate unique random positions for each goal
        for i in range(self.num_agents):
            self.goal_poses[i] = {
                "x": random.uniform(-1.5, 1.5),
                "y": random.uniform(-1.5, 1.5)
            }
        print("New goal poses:", self.goal_poses)

    def reset_simulation(self):
        req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_simulation_client.call_async(req)

    def delete_entities(self):
        for name in self.entity_names:
            req = DeleteEntity.Request()
            req.name = name
            while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.delete_entity_client.call_async(req)

    def spawn_entity(self, entity_name, pose):
        req = SpawnEntity.Request()
        req.name = entity_name
        req.xml = self.entity
        req.initial_pose = pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req)


def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    stage = args[0] if len(args) > 0 else "1"
    num_agents = int(args[1]) if len(args) > 1 else 1
    dqn_gazebo = DQNGazebo(stage, num_agents)
    rclpy.spin(dqn_gazebo)

    dqn_gazebo.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

