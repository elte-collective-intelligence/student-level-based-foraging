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

import collections
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.losses import MeanSquaredError

import json
import numpy
import os
import random
import sys
import time

import rclpy
from rclpy.node import Node

from turtlebot3_msgs.srv import Dqn

class DQNTest(Node):
    def __init__(self, stage, agent_count):
        super().__init__('dqn_test')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        # Stage
        self.stage = int(stage)

        # Number of agents
        self.agent_count = int(agent_count)

        # State size and action size
        self.state_size = 4
        self.action_size = 5
        self.episode_size = 3000

        # DQN hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64

        # Replay memory
        self.memory = collections.deque(maxlen=1000000)

        # Build model and target model for each agent
        self.models = [self.build_model() for _ in range(self.agent_count)]
        self.target_models = [self.build_model() for _ in range(self.agent_count)]

        # Load saved models
        self.load_model = True
        # self.load_episode = 1260
        self.load_episode = 320
        self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_dir_path = self.model_dir_path.replace(
            'turtlebot3_dqn/dqn_test',
            'model')

        if self.load_model:
            for i in range(self.agent_count):
                model_path = os.path.join(
                    self.model_dir_path,
                    f'stage{self.stage}_episode{self.load_episode}.h5')
                self.models[i].set_weights(load_model(model_path).get_weights())

                param_path = os.path.join(
                    self.model_dir_path,
                    f'stage{self.stage}_episode{self.load_episode}.json')
                with open(param_path) as outfile:
                    param = json.load(outfile)
                    self.epsilon = param.get('epsilon')

        """************************************************************
        ** Initialise ROS clients
        ************************************************************"""
        # Initialise clients for each agent
        self.dqn_com_clients = [
            self.create_client(Dqn, f'robot{i}/dqn_com') for i in range(self.agent_count)
        ]

        """************************************************************
        ** Start process
        ************************************************************"""
        self.process()

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def process(self):
        global_step = 0
        cumulative_reward_log_file = "/home/kwamboka/dqn_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/rewards.log"

        # Create or overwrite the cumulative reward log file
        with open(cumulative_reward_log_file, 'w') as log_file:
            log_file.write("Stage,Episode,Agent,Cumulative_Reward\n")

        for episode in range(self.load_episode + 1, self.episode_size):
            global_step += 1
            local_step = 0

            states = [[] for _ in range(self.agent_count)]
            next_states = [[] for _ in range(self.agent_count)]
            dones = [False] * self.agent_count
            inits = [True] * self.agent_count
            scores = [0] * self.agent_count  # To track cumulative rewards for agents

            self.get_logger().info(f"Starting episode {episode}...")

            # Reset DQN environment
            time.sleep(1.0)

            while not all(dones):
                local_step += 1

                for i in range(self.agent_count):
                    if dones[i]:
                        continue

                    # Action based on the current state
                    if local_step == 1:
                        action = 2  # Move forward
                    else:
                        states[i] = next_states[i]
                        action = int(self.get_action(states[i], i))

                    # Send action and receive next state and reward
                    req = Dqn.Request()
                    req.action = action
                    req.init = inits[i]

                    while not self.dqn_com_clients[i].wait_for_service(timeout_sec=1.0):
                        self.get_logger().info(f'Service for agent {i} not available, waiting again...')

                    future = self.dqn_com_clients[i].call_async(req)

                    while rclpy.ok():
                        rclpy.spin_once(self)
                        if future.done():
                            if future.result() is not None:
                                # Next state and reward
                                next_states[i] = future.result().state
                                reward = future.result().reward
                                dones[i] = future.result().done
                                scores[i] += reward  # Add to cumulative reward
                                inits[i] = False
                            else:
                                self.get_logger().error(
                                    f'Exception while calling service for agent {i}: {future.exception()}')
                            break

                    # While loop rate
                    time.sleep(0.01)

            # Log cumulative rewards after the episode ends
            with open(cumulative_reward_log_file, 'a') as log_file:
                for i in range(self.agent_count):
                    log_file.write(f"{self.stage},{episode},{i},{scores[i]}\n")

            self.get_logger().info(f"Episode {episode} completed. Cumulative Rewards: {scores}")

        self.get_logger().info(f"Cumulative reward logs saved to: {cumulative_reward_log_file}")




    def build_model(self):
        model = Sequential()
        model.add(Dense(
            64,
            input_shape=(self.state_size,),
            activation='relu',
            kernel_initializer='lecun_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss="mean_squared_error", optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def get_action(self, state, agent_idx):
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = numpy.asarray(state)
            q_value = self.models[agent_idx].predict(state.reshape(1, len(state)))
            return numpy.argmax(q_value[0])

def main(args=sys.argv[1:]):
    rclpy.init(args=args)

    if len(args) > 1:
        stage = args[0]
        agent_count = args[1]
    else:
        stage = 1 # Default stage value
        agent_count = 2 # Default agent count

    dqn_agent = DQNTest(stage, agent_count)
    rclpy.spin(dqn_agent)

    dqn_agent.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
