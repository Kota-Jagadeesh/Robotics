% main.m
% Main script for multi-objective trajectory optimization of a 2-DOF manipulator.

clear; clc; close all;

% Define parameters
l1 = 1;           % Length of first link (meters)
l2 = 1;           % Length of second link (meters)
target = [1.5, 0.5];  % Target position (meters)
numSteps = 50;    % Number of time steps
dt = 0.1;         % Time step (seconds)

fprintf('Creating RL Environment...\n');
% Create environment with parameters
env = createEnvironment(l1, l2, target, numSteps, dt);

fprintf('Validating Environment...\n');
% Validate environment
validateEnvironment(env);

fprintf('Creating PPO Agent...\n');
% Create PPO agent
agent = createPPOAgent(env);

fprintf('Training Agent...\n');
% Train agent (reduced episodes for testing)
% In main.m, use more stable training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 50, ...  % Reduced for submission
    'MaxStepsPerEpisode', numSteps, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', -1.0, ...  % Less strict stopping criterion
    'SaveAgentCriteria', 'AverageReward', ...
    'SaveAgentValue', -0.5);

trainResults = train(agent, env, trainOpts);

fprintf('Simulating and Plotting Results...\n');
% Simulate and visualize
simulateAndPlotTrajectory(agent, env, target, l1, l2, dt, 'optimized_trajectory.gif');

fprintf('Simulation completed successfully!\n');