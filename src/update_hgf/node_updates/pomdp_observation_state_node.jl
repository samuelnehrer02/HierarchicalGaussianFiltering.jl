###################################
######## Update prediction ########
###################################

##### Superfunction #####
function update_node_prediction!(node::PomdpObservationStateNode, stepsize::Real)

    # Update prediction
    node.states.prediction = calculate_prediction(node)

    return nothing
end

function calculate_prediction(node::PomdpObservationStateNode)

    # Taking the observation likelihood arrays from each of the OL parents
    parent_predictions = map(x -> x.states.prediction, collect(values(node.edges.pomdp_observation_parents)))
    
    updated_A = parent_predictions

    return updated_A

end

##################################
######## Update posterior ########
##################################

##### Superfunction #####
"""
    update_node_posterior!(node::PomdpStateNode)

Update the posterior of the PomdpStateNode, which distributes the B-matrices to the correct
TPMNodes.
"""
function update_node_posterior!(node::PomdpObservationStateNode, update_type::EnhancedUpdate)

    #Update posterior 
    node.states.posterior = calculate_posterior(node)

    return nothing
end


function calculate_posterior(node::PomdpObservationStateNode)

    # Extract the pomdp input child
    child = node.edges.observation_children[1]

    # Exatracting observation from input node
    observation = child.states.observation

    # Number of modalities
    n_modalities = length(observation)

    # Extracting the number of observations from input node
    n_observations = node.states.n_observations
    n_observations = Vector(Int64.(n_observations))

    # Process the observation for creation of the observation likelihood arrays
    processed_observation = process_observation(observation, n_modalities, n_observations)

    # Extract the child's qs_current for creating the posterior of the OL nodes
    qs = child.states.qs_current
    qs_cross = outer_product(qs)

    # Observation likelihood arrays (ola)
    ola = Vector{Any}(undef, n_modalities)

    for modality in 1:n_modalities
        ola[modality] = outer_product(processed_observation[modality], qs_cross)
    end

    posterior = ola
    
    return posterior
end


function update_node_value_prediction_error!(node::PomdpObservationStateNode)
    return nothing
end

function update_node_precision_prediction_error!(node::PomdpObservationStateNode)
    return nothing
end

