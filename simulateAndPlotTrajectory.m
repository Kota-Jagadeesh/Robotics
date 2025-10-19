% simulateAndPlotTrajectory.m
% Function to simulate the trained agent and plot the trajectory.

function simulateAndPlotTrajectory(agent, env, target, l1, l2, dt, gifFileName)
    % Initialize simulation parameters
    numSteps = 50;
    stateHistory = zeros(numSteps, 4);
    positionHistory = zeros(numSteps, 2);
    energyHistory = zeros(numSteps, 1);
    distanceHistory = zeros(numSteps, 1);
    rewardHistory = zeros(numSteps, 1);

    % Reset environment
    initialObs = reset(env);
    fullObs = initialObs;
    stepCount = 0;

    % Simulate
    figure('Name', 'Trajectory Simulation');
    hold on;

    while stepCount < numSteps
        % Get action from agent
        action = getAction(agent, fullObs);
        
        % Ensure action is 2x1 vector
        if numel(action) == 1
            action = [action; 0]; % Make it 2x1 by adding zero for second joint
        elseif size(action, 1) ~= 2
            action = action(1:2); % Take first 2 elements
            if length(action) < 2
                action = [action(1); 0];
            end
        end
        
        % Step through environment
        [nextObs, reward, isDone, ~] = step(env, action);
        
        % Extract state for plotting
        currentState = fullObs(1:4);
        x = l1 * cos(currentState(1)) + l2 * cos(currentState(1) + currentState(2));
        y = l1 * sin(currentState(1)) + l2 * sin(currentState(1) + currentState(2));
        
        % Store data
        stateHistory(stepCount + 1, :) = currentState';
        positionHistory(stepCount + 1, :) = [x, y];
        energyHistory(stepCount + 1) = sum(action.^2);
        distanceHistory(stepCount + 1) = norm([x, y] - target);
        rewardHistory(stepCount + 1) = reward;
        
        % Update observation
        fullObs = nextObs;
        stepCount = stepCount + 1;
        
        % Plot
        plot(x, y, 'b.', 'MarkerSize', 10);
        drawnow;
        
        if isDone
            break;
        end
    end
    
    hold off;

    % Plot results
    figure('Name', 'PPO Simulation Results', 'Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    plot(0:stepCount-1, stateHistory(1:stepCount, 1), 'r-', 'LineWidth', 2);
    hold on;
    plot(0:stepCount-1, stateHistory(1:stepCount, 2), 'g-', 'LineWidth', 2);
    title('Joint Angles');
    xlabel('Step'); ylabel('Angle (rad)');
    legend('\theta_1', '\theta_2');
    grid on;

    subplot(2, 3, 2);
    plot(positionHistory(1:stepCount, 1), positionHistory(1:stepCount, 2), 'b-', 'LineWidth', 2);
    hold on;
    plot(target(1), target(2), 'ro', 'MarkerSize', 10);
    plot(0, 0, 'ko', 'MarkerSize', 8);
    title('End-Effector Trajectory');
    xlabel('X (m)'); ylabel('Y (m)');
    legend('Path', 'Target', 'Base');
    axis equal; grid on;

    subplot(2, 3, 3);
    plot(0:stepCount-1, energyHistory(1:stepCount), 'm-', 'LineWidth', 2);
    title('Energy Consumption');
    xlabel('Step'); ylabel('Energy');
    grid on;

    subplot(2, 3, 4);
    plot(0:stepCount-1, distanceHistory(1:stepCount), 'c-', 'LineWidth', 2);
    title('Distance to Target');
    xlabel('Step'); ylabel('Distance (m)');
    grid on;

    subplot(2, 3, 5);
    plot(0:stepCount-1, rewardHistory(1:stepCount), 'g-', 'LineWidth', 2);
    title('Reward');
    xlabel('Step'); ylabel('Reward');
    grid on;

    subplot(2, 3, 6);
    plot(0:stepCount-1, cumsum(rewardHistory(1:stepCount)), 'b-', 'LineWidth', 2);
    title('Cumulative Reward');
    xlabel('Step'); ylabel('Cumulative Reward');
    grid on;

    % Save figure
    saveas(gcf, 'simulation_results.png');

    % Create GIF
    figure('Position', [200, 200, 800, 600]);
    for i = 1:stepCount
        cla; hold on; axis equal; xlim([-2 2]); ylim([-2 2]);
        
        % Plot target and base
        plot(target(1), target(2), 'g*', 'MarkerSize', 15);
        plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        
        % Plot trajectory
        plot(positionHistory(1:i, 1), positionHistory(1:i, 2), 'b--');
        
        % Plot current position
        currentState = stateHistory(i, :);
        x1 = l1 * cos(currentState(1));
        y1 = l1 * sin(currentState(1));
        x2 = x1 + l2 * cos(currentState(1) + currentState(2));
        y2 = y1 + l2 * sin(currentState(1) + currentState(2));
        
        % Plot links
        plot([0 x1], [0 y1], 'r-', 'LineWidth', 3);
        plot([x1 x2], [y1 y2], 'r-', 'LineWidth', 3);
        plot(x2, y2, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        
        title(sprintf('Step %d/%d', i, stepCount));
        xlabel('X (m)'); ylabel('Y (m)'); grid on;
        drawnow;
        
        % Save frame
        frame = getframe(gcf);
        im = frame2im(frame);
        [A, map] = rgb2ind(im, 256);
        if i == 1
            imwrite(A, map, gifFileName, 'LoopCount', inf, 'DelayTime', 0.1);
        else
            imwrite(A, map, gifFileName, 'WriteMode', 'append', 'DelayTime', 0.1);
        end
    end
    close(gcf);

    % Summary
    fprintf('\n=== SIMULATION SUMMARY ===\n');
    fprintf('Total Energy: %.2f\n', sum(energyHistory(1:stepCount)));
    fprintf('Final Distance: %.2f m\n', distanceHistory(stepCount));
    fprintf('Total Reward: %.2f\n', sum(rewardHistory(1:stepCount)));
    fprintf('========================\n');
end