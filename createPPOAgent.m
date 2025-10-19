function agent = createPPOAgent(env)
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    
    sharedLayers = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')];
    
    meanLayers = fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'mean_output');
    
    logStdLayers = [
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'logstd_fc', ...
            'Bias', zeros(actInfo.Dimension(1),1), ...
            'Weights', 0.01*randn(actInfo.Dimension(1), 64)) ...
        softplusLayer('Name', 'logstd_output')];
    
    actorLayers = layerGraph(sharedLayers);
    actorLayers = addLayers(actorLayers, meanLayers);
    actorLayers = addLayers(actorLayers, logStdLayers);
    actorLayers = connectLayers(actorLayers, 'relu2', 'mean_output');
    actorLayers = connectLayers(actorLayers, 'relu2', 'logstd_fc');
    
    criticNetwork = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(1, 'Name', 'output')];
    
    actorNet = dlnetwork(actorLayers);
    criticNet = dlnetwork(criticNetwork);
    
    actor = rlContinuousGaussianActor(actorNet, obsInfo, actInfo, ...
        'ActionMeanOutputNames', 'mean_output', ...
        'ActionStandardDeviationOutputNames', 'logstd_output');
    
    critic = rlValueFunction(criticNet, obsInfo);
    
    agentOpts = rlPPOAgentOptions(...
        'ExperienceHorizon', 512, ...
        'AdvantageEstimateMethod', 'gae', ...
        'GAEFactor', 0.95, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 32, ...
        'NumEpoch', 5, ...
        'ClipFactor', 0.2, ...
        'EntropyLossWeight', 0.02);
    
    agent = rlPPOAgent(actor, critic, agentOpts);
end