# 2-DOF Manipulator Trajectory Optimization (PPO-RL)

This project utilizes **Reinforcement Learning (RL)** — specifically the **Proximal Policy Optimization (PPO)** algorithm — to compute the optimal control strategy (torques) for a **two-degree-of-freedom (2-DOF)** robotic arm.  
The goal is to efficiently move the manipulator's end-effector from its starting point to a target position while balancing multiple performance metrics.

---

## What This Project Does ?

The core function is to optimize the robot's movement based on a **multi-objective reward function** that aims to:

-  **Minimize Distance:** Reach the target location accurately.  
-  **Minimize Energy:** Reduce the magnitude of the applied joint torques.  
-  **Minimize Time:** Complete the task within a limited number of steps.  
-  **Maximize Smoothness:** Maintain stable motion by penalizing excessive changes in acceleration (jerk).

---

## How It Works (Code Structure)

The solution is implemented in **MATLAB** using the **Reinforcement Learning Toolbox™** and organized into four main files:

| File | Role | Description |
|------|------|--------------|
| `createEnvironment.m` | RL Environment | Defines the operational space (state and action limits), the kinematic and simplified dynamic equations for the 2-DOF arm, and implements the multi-objective reward function. Includes numerical stability checks (`wrapToPi`, `isfinite`). |
| `createPPOAgent.m` | PPO Agent Configuration | Sets up the PPO algorithm, configuring the Actor (policy network for torque output) and the Critic (value network for estimating expected returns) using two interconnected neural networks. |
| `main.m` | Execution Script | Defines the physical parameters (link lengths, target), validates the environment, creates the agent, and executes the primary training loop using specified options (`trainOpts`). |
| `simulateAndPlotTrajectory.m` | Simulation & Visualization | After training, this script runs the final agent to generate the optimized trajectory. It handles action output formats (cell-to-numeric conversion, size checks) and produces detailed plots and an animated GIF of the robot's path. |

---

## Running the Project

1. Ensure **MATLAB** with the **Reinforcement Learning Toolbox™** is installed.  
2. Place all project files in the same directory.  
3. Execute:

   ```matlab
   main
   ```
4. The script will:

   - Initiate training.
    - Save the results and visualizations in the working directory.
