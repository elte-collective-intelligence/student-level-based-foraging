# Level-Based Foraging (LBF): Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-level-based-foraging/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-level-based-foraging/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-level-based-foraging/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-level-based-foraging/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-level-based-foraging/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-level-based-foraging)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](LICENSE)

## About the project

# Multi-Agent Communication with PPO (TorchRL + Jumanji)

The aim of the project is learned communication between cooperative agents in a multi-agent reinforcement learning setting. Using PPO and TorchRL, agents with and without communication are compared in the Level-Based Foraging environment.

The goal is to implement and test communication in this environment and to see how it helps he agents cooperate in the task. 


## Overview

- **Environment**: `LevelBasedForaging-v0` (2 agents)
- **Algorithms**:
  - **MAPPO** (centralized critic)
  - **IPPO** (independent critics)
- **Communication**:
  - Discrete symbols with a fixed vocabulary size, messages are appended to observations in the next time step

## Experiments

Three configurations are being tested with both MAPPO and IPPO

**- No Communication**

**- Communication with a vocabulary size of 4**

**- Communication with a vocabulary size of 1 (to compare with no communication)**


## What is implemented

- Multi-agent PPO training loop using TorchRL
- Discrete communication channel
- Message passing via agent-to-agent swapping
- Centralized and decentralized critics
- Logging and visualization of:
  - Episode reward curves
  - Message entropy over training
  - Message token usage distributions


## Results

The results of the experiments show that the agents have not improved at all with communication during and after training. This is possibly because of an undetected fault in the code that breaks the learning process.

## Recreation of results

Currently, because of some limitaions the notebook can only be run in **Google Colab**.
