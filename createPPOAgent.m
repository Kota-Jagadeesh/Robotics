% createPPOAgent.m
% Function to create and configure PPO agent with stable standard deviation.

function agent = createPPOAgent(env)
    % Get observation and action info
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    
    % Shared layers for actor network
    sharedLayers = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')];
    
    % Branch for mean output
    meanLayers = fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'mean_output');
    
    % Branch for log-std output with softplus activation for stability
    logStdLayers = [
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'logstd_fc', ...
            'Bias', zeros(actInfo.Dimension(1),1), ...  % Initialize bias to 0
            'Weights', 0.01*randn(actInfo.Dimension(1), 64)) ...  % Small random weights
        softplusLayer('Name', 'logstd_output')];  % Ensure nonnegative output
    
    % Combine into a single layer graph
    actorLayers = layerGraph(sharedLayers);
    actorLayers = addLayers(actorLayers, meanLayers);
    actorLayers = addLayers(actorLayers, logStdLayers);
    actorLayers = connectLayers(actorLayers, 'relu2', 'mean_output');
    actorLayers = connectLayers(actorLayers, 'relu2', 'logstd_fc');  % Connect to log-std FC layer
    
    % Analyze and plot the network (optional, for debugging)
    % analyzeNetwork(actorLayers);
    
    % Critic network (value function)
    criticNetwork = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'output')];
    
    % Convert to dlnetwork
    actorNet = dlnetwork(actorLayers);
    criticNet = dlnetwork(criticNetwork);

    % Create actor and critic representations
    % Map 'mean_output' for action mean and 'logstd_output' for standard deviation
    actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
        'ActionMeanOutputNames', 'mean_output', ...
        'ActionStandardDeviationOutputNames', 'logstd_output');
    
    critic = rlValueFunction(criticNet, obsInfo);
    
    % Create PPO agent options with conservative settings for stability
    agentOpts = rlPPOAgentOptions(...
        'ExperienceHorizon', 512, ...  % Reduced for stability
        'AdvantageEstimateMethod', 'gae', ...
        'GAEFactor', 0.95, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 32, ...  % Reduced batch size
        'NumEpoch', 5, ...  % Reduced epochs
        'ClipFactor', 0.2, ...
        'EntropyLossWeight', 0.02);  % Increased for exploration
    
    % Create PPO agent
    agent = rlPPOAgent(actor, critic, agentOpts);
end