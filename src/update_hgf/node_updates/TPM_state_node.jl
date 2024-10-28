###################################
######## Update prediction ########
###################################

##### Superfunction #####
function update_node_prediction!(node::TPMStateNode, stepsize::Real)

    # Update prediction
    node.states.prediction = calculate_prediction(node)

    return nothing
end

function calculate_prediction(node::TPMStateNode)

    #Get current parent predictions
    parent_predictions = map(x -> x.states.prediction, collect(values(node.edges.tpm_parents)))
    
    # Convert to matrix
    prediction_matrix = Matrix(hcat(parent_predictions...))

    return prediction_matrix
end

##################################
######## Update posterior ########
##################################

##### Superfunction #####
"""
    update_node_posterior!(node::TPMStateNode)

Update the posterior of a single Transition Probability Matrix state node.
"""
function update_node_posterior!(node::TPMStateNode, update_type::ClassicUpdate)

    #Update posterior 
    node.states.posterior = calculate_posterior(node)

    return nothing
end

function calculate_posterior(node::TPMStateNode)

    # Extract the pomdp child
    child = node.edges.pomdp_children[1]

    # From the name, extracts the factor and action.
    # If action and factor is 1, perform update, otherwise posterior is equal to missing
    node_name = node.name
    m_f = match(r"f(\d+)", node_name)
    n_f = parse(Int, m_f.captures[1])

    m_a = match(r"a(\d+)", node_name)
    n_a = parse(Int, m_a.captures[1])

    action_vector = child.states.posterior_policy

    if !ismissing(action_vector[n_f][n_a])

        # Initialize previous input
        previous_input = node.states.previous_qs

        # Initialize input as previous input
        input = deepcopy(previous_input)

        # Update the input from the observation child node
        input .= child.states.posterior[n_f]

        # Initialize posterior as previous posterior
        posterior = input .* input'
        
        # Calculate the posterior as an outer product of the previous and current input 
        posterior .= previous_input .* input'

        
    else
        posterior = missing
    end

    # Setting previous_qs for next calculation, regardless of chosen action
    node.states.previous_qs .= child.states.posterior[n_f]

    return posterior
end

# function calculate_posterior(node::TPMStateNode)

#     # Extract the pomdp child
#     child = node.edges.observation_children[1]

#     # Initialize previous input
#     previous_input = node.states.previous_qs

#     # Initialize input as previous input
#     input = deepcopy(previous_input)

#     # Update the input from the observation child node
#     input .= child.states.input_value

#     # Initialize posterior as previous posterior
#     posterior = node.states.posterior
    
#     # Calculate the posterior as an outer product of the previous and current input 
#     posterior .= previous_input .* input'

#     # And save the input as previous input for the next iteration
#     node.states.previous_qs .= input

#     return posterior
# end


function update_node_value_prediction_error!(node::TPMStateNode)
    return nothing
end

function update_node_precision_prediction_error!(node::TPMStateNode)
    return nothing
end

