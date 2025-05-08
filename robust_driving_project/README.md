# Robust Autonomous Driving Project

This project implements a robust decision-making system for autonomous driving using highway-env and Soft Actor-Critic (SAC) algorithm with uncertainty estimation.

## Features

- SAC-based autonomous driving agent
- Uncertainty estimation using Q-network ensemble
- Fallback policy using IDM + MOBIL
- Comprehensive evaluation metrics

## Installation

1. Install required packages:
```bash
pip install gym highway-env stable-baselines3 torch numpy