




##################################
######## Update posterior ########
##################################

##### Superfunction #####
"""
    update_node_posterior!(node::PomdpStateNode)

Update the posterior of the PomdpStateNode, which distributes the B-matrices to the correct
TPMNodes.
"""
function update_node_posterior!(node::PomdpStateNode, update_type::ClassicUpdate)

    #Update posterior 
    node.states.posterior, node.states.posterior_policy = calculate_posterior(node)

    return nothing
end


function calculate_posterior(node::PomdpStateNode)

    # Extract the pomdp input child
    child = node.edges.pomdp_children[1]

    # Create previous qs
    node.states.previous_qs = node.states.posterior

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


