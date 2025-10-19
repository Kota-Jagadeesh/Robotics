function simulateAndPlotTrajectory(agent, env, target, l1, l2, dt, gifFileName)
    numSteps = 50;
    stateHistory = zeros(numSteps, 4);
    positionHistory = zeros(numSteps, 2);
    energyHistory = zeros(numSteps, 1);
    distanceHistory = zeros(numSteps, 1);
    rewardHistory = zeros(numSteps, 1);
    
    initialObs = reset(env);
    fullObs = initialObs;
    stepCount = 0;
    
    figure('Name', 'Trajectory Simulation');
    hold on;
    
    while stepCount < numSteps
        action = getAction(agent, fullObs);
        
        if iscell(action)
            action = action{1};
        end
        
        action = double(action);
        
        if isrow(action)
            action = action';
        end
        
        if size(action, 1) < 2
            action = [action; zeros(2 - size(action, 1), 1)];
        elseif size(action, 1) > 2
             action = action(1:2);
        end
        
        [nextObs, reward, isDone, ~] = step(env, action);
        
        currentState = fullObs(1:4);
        theta1 = currentState(1);
        theta2 = currentState(2);
        
        x = l1 * cos(theta1) + l2 * cos(theta1 + theta2);
        y = l1 * sin(theta1) + l2 * sin(theta1 + theta2);
        
        stepCount = stepCount + 1;
        stateHistory(stepCount, :) = currentState';
        positionHistory(stepCount, :) = [x, y];
        energyHistory(stepCount) = sum(action.^2);
        distanceHistory(stepCount) = norm([x, y] - target);
        rewardHistory(stepCount) = reward;
        
        fullObs = nextObs;
        
        plot(x, y, 'b.', 'MarkerSize', 10);
        drawnow;
        
        if isDone
            break;
        end
    end
    
    stateHistory = stateHistory(1:stepCount, :);
    positionHistory = positionHistory(1:stepCount, :);
    energyHistory = energyHistory(1:stepCount);
    distanceHistory = distanceHistory(1:stepCount);
    rewardHistory = rewardHistory(1:stepCount);
    
    hold off;
    
    figure('Name', 'PPO Simulation Results', 'Position', [100, 100, 1200, 800]);
    
    subplot(2, 3, 1);
    plot(0:stepCount-1, stateHistory(:, 1), 'r-', 'LineWidth', 2);
    hold on;
    plot(0:stepCount-1, stateHistory(:, 2), 'g-', 'LineWidth', 2);
    title('Joint Angles');
    xlabel('Step'); ylabel('Angle (rad)');
    legend('\theta_1', '\theta_2');
    grid on;
    
    subplot(2, 3, 2);
    plot(positionHistory(:, 1), positionHistory(:, 2), 'b-', 'LineWidth', 2);
    hold on;
    plot(target(1), target(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    title('End-Effector Trajectory');
    xlabel('X (m)'); ylabel('Y (m)');
    legend('Path', 'Target', 'Base');
    axis equal; 
    xlim([-l1-l2 l1+l2]); ylim([-l1-l2 l1+l2]);
    grid on;
    
    subplot(2, 3, 3);
    plot(0:stepCount-1, energyHistory, 'm-', 'LineWidth', 2);
    title('Energy Consumption');
    xlabel('Step'); ylabel('Energy ($\tau^2$)', 'Interpreter', 'latex');
    grid on;
    
    subplot(2, 3, 4);
    plot(0:stepCount-1, distanceHistory, 'c-', 'LineWidth', 2);
    title('Distance to Target');
    xlabel('Step'); ylabel('Distance (m)');
    grid on;
    
    subplot(2, 3, 5);
    plot(0:stepCount-1, rewardHistory, 'g-', 'LineWidth', 2);
    title('Reward per Step');
    xlabel('Step'); ylabel('Reward');
    grid on;
    
    subplot(2, 3, 6);
    plot(0:stepCount-1, cumsum(rewardHistory), 'b-', 'LineWidth', 2);
    title('Cumulative Reward');
    xlabel('Step'); ylabel('Cumulative Reward');
    grid on;
    
    saveas(gcf, 'simulation_results.png');
    
    figure('Position', [200, 200, 800, 600]);
    for i = 1:stepCount
        cla; hold on; axis equal; 
        xlim([-l1-l2 l1+l2]); ylim([-l1-l2 l1+l2]);
        
        plot(target(1), target(2), 'g*', 'MarkerSize', 15);
        plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        
        plot(positionHistory(1:i, 1), positionHistory(1:i, 2), 'b--');
        
        currentState = stateHistory(i, :);
        theta1 = currentState(1);
        theta2 = currentState(2);
        
        x1 = l1 * cos(theta1);
        y1 = l1 * sin(theta1);
        x2 = l1 * cos(theta1) + l2 * cos(theta1 + theta2);
        y2 = l1 * sin(theta1) + l2 * sin(theta1 + theta2);
        
        plot([0 x1], [0 y1], 'r-', 'LineWidth', 3);
        plot(x1, y1, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', [0.8 0 0]);
        plot([x1 x2], [y1 y2], 'r-', 'LineWidth', 3);
        plot(x2, y2, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        
        title(sprintf('Optimized Trajectory - Step %d/%d', i, numSteps));
        xlabel('X (m)'); ylabel('Y (m)'); grid on;
        drawnow;
        
        frame = getframe(gcf);
        im = frame2im(frame);
        [A, map] = rgb2ind(im, 256);
        delay = 0.1;
        if i == 1
            imwrite(A, map, gifFileName, 'LoopCount', inf, 'DelayTime', delay);
        else
            imwrite(A, map, gifFileName, 'WriteMode', 'append', 'DelayTime', delay);
        end
    end
    close(gcf);
    
    fprintf('\n=== SIMULATION SUMMARY ===\n');
    fprintf('Total Steps: %d\n', stepCount);
    fprintf('Total Energy (Sum of $\\tau^2$): %.2f\n', sum(energyHistory));
    fprintf('Final Distance to Target: %.4f m\n', distanceHistory(end));
    fprintf('Total Cumulative Reward: %.2f\n', sum(rewardHistory));
    fprintf('========================\n');
end