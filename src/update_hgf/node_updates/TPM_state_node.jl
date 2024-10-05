##################################
######## Update posterior ########
##################################

##### Superfunction #####
"""
    update_node_posterior!(node::TPMStateNode)

Update the posterior of a single Transition Probability Matrix state node.
"""
function update_node_posterior!(node::HierarchicalGaussianFiltering.TPMStateNode, update_type::ClassicUpdate)

    #Update posterior 
    node.states.posterior = calculate_posterior(node)

    return nothing
end



function calculate_posterior(node::HierarchicalGaussianFiltering.TPMStateNode)

    # Extract the pomdp child
    child = node.edges.observation_children[1]

    # Initialize previous input
    previous_input = node.states.previous_qs

    # Initialize input as previous input
    input = deepcopy(previous_input)

    # Update the input from the observation child node
    input .= child.states.input_value

    # Initialize posterior as previous posterior
    posterior = node.states.posterior
    
    # Calculate the posterior as an outer product of the previous and current input 
    posterior .= previous_input .* input'

    # And save the input as previous input for the next iteration
    node.states.previous_qs .= input

    return posterior
end


