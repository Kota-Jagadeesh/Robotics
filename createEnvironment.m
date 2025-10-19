% createEnvironment.m
% Function to create RL environment for 2-DOF manipulator.

function env = createEnvironment(l1, l2, target, numSteps, dt)
    % Define observation space: [theta1, theta2, theta1dot, theta2dot, stepCount]
    obsInfo = rlNumericSpec([5 1]);
    obsInfo.LowerLimit = [-pi; -pi; -5; -5; 0];
    obsInfo.UpperLimit = [pi; pi; 5; 5; numSteps];
    obsInfo.Name = 'Robot State';
    obsInfo.Description = 'theta1, theta2, theta1dot, theta2dot, stepCount';

    % Define action space: [torque1, torque2]
    actInfo = rlNumericSpec([2 1]);
    actInfo.LowerLimit = [-2; -2];
    actInfo.UpperLimit = [2; 2];
    actInfo.Name = 'Torques';
    actInfo.Description = 'Joint torques';

    % Create environment with step and reset functions (only 4 arguments)
    env = rlFunctionEnv(obsInfo, actInfo, @stepFunction, @resetFunction);

    % Note: No direct property assignment to env; parameters are handled in resetFunction
end

% Reset function - Returns exactly ONE output matching obsInfo and initializes loggedSignals
function [initialObs, loggedSignals] = resetFunction()
    % Initial state: [theta1, theta2, theta1dot, theta2dot, stepCount]
    initialObs = [0; 0; 0; 0; 0];  % Single 5x1 vector output

    % Initialize logged signals with environment parameters
    loggedSignals = struct(...
        'Environment', struct('l1', 1, 'l2', 1, 'target', [1.5, 0.5], 'numSteps', 50, 'dt', 0.1), ...
        'StateHistory', [], ...
        'PositionHistory', [], ...
        'EnergyHistory', [], ...
        'DistanceHistory', [], ...
        'RewardHistory', [], ...
        'Observation', initialObs);
end

% Step function - Returns exactly 4 outputs
function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, loggedSignals)
    % Extract current state from the observation within loggedSignals
    currentObs = loggedSignals.Observation;
    theta1 = currentObs(1);
    theta2 = currentObs(2);
    theta1dot = currentObs(3);
    theta2dot = currentObs(4);
    stepCount = currentObs(5);

    % Get environment parameters from loggedSignals
    env = loggedSignals.Environment;
    l1 = env.l1;
    l2 = env.l2;
    target = env.target;
    numSteps = env.numSteps;
    dt = env.dt;

    % Update step count
    stepCount = stepCount + 1;

    % Simple dynamics update (torque to acceleration)
    theta1_ddot = action(1);  % torque1 -> angular acceleration1
    theta2_ddot = action(2);  % torque2 -> angular acceleration2

    % Integrate to get new velocities and positions
    theta1dot = theta1dot + theta1_ddot * dt;
    theta2dot = theta2dot + theta2_ddot * dt;
    theta1 = theta1 + theta1dot * dt + 0.5 * theta1_ddot * dt^2;  % Include acceleration term
    theta2 = theta2 + theta2dot * dt + 0.5 * theta2_ddot * dt^2;

    % Constrain joint angles
    theta1 = max(-pi, min(pi, theta1));
    theta2 = max(-pi, min(pi, theta2));

    % Compute end-effector position using forward kinematics
    x = l1 * cos(theta1) + l2 * cos(theta1 + theta2);
    y = l1 * sin(theta1) + l2 * sin(theta1 + theta2);

    % Calculate distance to target
    distance = sqrt((x - target(1))^2 + (y - target(2))^2);

    % Multi-objective reward function
    reward_distance = -distance;           % Minimize distance to target
    reward_energy = -0.1 * (action(1)^2 + action(2)^2);  % Penalize energy (torque squared)
    reward_time = -0.01 * stepCount * dt;  % Penalize time
    reward_smoothness = -0.05 * (theta1_ddot^2 + theta2_ddot^2);  % Penalize jerk

    reward = reward_distance + reward_energy + reward_time + reward_smoothness;

    % Termination conditions
    isDone = (distance < 0.1) || (stepCount >= numSteps);

    % Next observation
    nextObs = [theta1; theta2; theta1dot; theta2dot; stepCount];

    % Update logged signals
    loggedSignals.StateHistory = [loggedSignals.StateHistory; [theta1, theta2, theta1dot, theta2dot]];
    loggedSignals.PositionHistory = [loggedSignals.PositionHistory; [x, y]];
    loggedSignals.EnergyHistory = [loggedSignals.EnergyHistory; action(1)^2 + action(2)^2];
    loggedSignals.DistanceHistory = [loggedSignals.DistanceHistory; distance];
    loggedSignals.RewardHistory = [loggedSignals.RewardHistory; reward];
    loggedSignals.Observation = nextObs;  % Update observation for next step
    loggedSignals.Environment = env;     % Preserve environment parameters
end