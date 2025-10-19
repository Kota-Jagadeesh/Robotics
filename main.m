clear; clc; close all;
l1 = 1;
l2 = 1;
target = [1.5, 0.5];
numSteps = 50;
dt = 0.1;

fprintf('Creating RL Environment...\n');
env = createEnvironment(l1, l2, target, numSteps, dt);

fprintf('Validating Environment...\n');
validateEnvironment(env); 

fprintf('Creating PPO Agent...\n');
agent = createPPOAgent(env);

fprintf('Training Agent...\n');
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 50, ...
    'MaxStepsPerEpisode', numSteps, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', -1.0, ...
    'SaveAgentCriteria', 'AverageReward', ...
    'SaveAgentValue', -0.5);
trainResults = train(agent, env, trainOpts);

fprintf('Simulating and Plotting Results...\n');
simulateAndPlotTrajectory(agent, env, target, l1, l2, dt, 'optimized_trajectory.gif');

fprintf('Simulation completed successfully!\n');