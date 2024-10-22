###################################
######## Update prediction ########
###################################

##### Superfunction #####
function update_node_prediction!(node::PomdpStateNode, stepsize::Real)

    # Update prediction
    node.states.prediction = calculate_prediction(node)

    return nothing
end

function calculate_prediction(node::PomdpStateNode)

    #Get current parent predictions
    parent_predictions = map(x -> x.states.prediction, collect(values(node.edges.pomdp_parents)))
    
    # Create correct array for B-matrix
    n_control = node.states.n_control
    n_states = node.states.n_states

    B_matrix = [zeros(n_states[i], n_states[i], n_control[i]) for i in 1:length(n_control)]

    # Populating the B-matrix with the correct predictions
    iteration = 0

    for i in 1:length(n_control)
        for j in 1:n_control[i]
            iteration += 1
            B_matrix[i][:,:,j] = parent_predictions[iteration]
        end
    end

    return B_matrix

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
function update_node_posterior!(node::PomdpStateNode, update_type::EnhancedUpdate)

    #Update posterior 
    node.states.posterior, node.states.posterior_policy = calculate_posterior(node)

    return nothing
end


function calculate_posterior(node::PomdpStateNode)

    # Extract the pomdp input child
    child = node.edges.pomdp_children[1]

    # Create previous qs
    # node.states.previous_qs = node.states.posterior

    # Creating the posterior over actions as a one-hot encoded vector, with missing values for 0
    n_factors = length(node.states.n_control)
    n_control = node.states.n_control
    action = child.states.policy_chosen

    one_hot_action = Vector{Vector{Union{Missing, Int}}}(undef, n_factors)

    for i in 1:n_factors

        one_hot_action[i] = Vector{Union{Missing, Int}}(missing, n_control[i])

        one_hot_action[i][action[i]] = 1

    end

    # Add the one-hot encoded action to the posterior
    posterior_policy = one_hot_action

    # Extract the input (qs) and the action chosen from PomdpInputNode
    posterior = child.states.input_value
    
    return posterior, posterior_policy
end


function update_node_value_prediction_error!(node::PomdpStateNode)
    return nothing
end

function update_node_precision_prediction_error!(node::PomdpStateNode)
    return nothing
end

