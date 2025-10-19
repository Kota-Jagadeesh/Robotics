function env = createEnvironment(l1, l2, target, numSteps, dt)
    obsInfo = rlNumericSpec([5 1]);
    obsInfo.LowerLimit = [-pi; -pi; -5; -5; 0];
    obsInfo.UpperLimit = [pi; pi; 5; 5; numSteps];
    obsInfo.Name = 'Robot State';
    obsInfo.Description = 'theta1, theta2, theta1dot, theta2dot, stepCount';

    actInfo = rlNumericSpec([2 1]);
    actInfo.LowerLimit = [-2; -2];
    actInfo.UpperLimit = [2; 2];
    actInfo.Name = 'Torques';
    actInfo.Description = 'Joint torques';

    stepHandle = @(action, loggedSignals) stepFunction(action, loggedSignals, l1, l2, target, numSteps, dt);
    
    env = rlFunctionEnv(obsInfo, actInfo, stepHandle, @resetFunction);
end

function [initialObs, loggedSignals] = resetFunction()
    initialObs = [0; 0; 0; 0; 0]; 
    loggedSignals = struct('Observation', initialObs);
end

function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, loggedSignals, l1, l2, target, numSteps, dt)
    
    currentObs = loggedSignals.Observation;
    theta1 = currentObs(1);
    theta2 = currentObs(2);
    theta1dot = currentObs(3);
    theta2dot = currentObs(4);
    stepCount = currentObs(5);
    
    dt = double(dt); 

    stepCount = stepCount + 1;

    theta1_ddot = action(1);
    theta2_ddot = action(2);

    theta1dot = theta1dot + theta1_ddot * dt;
    theta2dot = theta2dot + theta2_ddot * dt;

    theta1 = theta1 + theta1dot * dt + 0.5 * theta1_ddot * dt^2;
    theta2 = theta2 + theta2dot * dt + 0.5 * theta2_ddot * dt^2;
    
    theta1 = wrapToPi(theta1);
    theta2 = wrapToPi(theta2);

    max_vel = 5;
    theta1dot = max(-max_vel, min(max_vel, theta1dot));
    theta2dot = max(-max_vel, min(max_vel, theta2dot));

    x = l1 * cos(theta1) + l2 * cos(theta1 + theta2);
    y = l1 * sin(theta1) + l2 * sin(theta1 + theta2);

    distance = norm([x, y] - target);

    reward_distance = -distance;           
    reward_energy = -0.1 * (action(1)^2 + action(2)^2);
    reward_time = -0.01 * stepCount * dt;
    reward_smoothness = -0.05 * (theta1_ddot^2 + theta2_ddot^2);
    
    reward = reward_distance + reward_energy + reward_time + reward_smoothness;

    if ~isfinite(reward)
        warning('RL Environment: Reward became NaN or Inf. Penalizing and terminating.');
        reward = -1000;
        isDone = true; 
    else
        isDone = (distance < 0.1) || (stepCount >= numSteps);
    end

    nextObs = [theta1; theta2; theta1dot; theta2dot; stepCount];

    loggedSignals.Observation = nextObs; 
end