###################################
######## Update prediction ########
###################################

##### Superfunction #####
function update_node_prediction!(node::OLStateNode, stepsize::Real)

    # Update prediction
    node.states.prediction = calculate_prediction(node)

    return nothing
end

function calculate_prediction(node::HierarchicalGaussianFiltering.OLStateNode)

    # Extract the pomdp observation child
    child = node.edges.pomdp_observation_children[1]

    #Get current parent predictions
    parent_predictions = map(x -> x.states.prediction, collect(values(node.edges.ol_parents)))
    
    # Taking the n_observations and n_states from the child node
    n_observations = child.states.n_observations
    n_states = child.states.n_states

    # Extract the modality of this node from name
    modality = match(r"m(\d+)", node.name)
    modality = parse(Int, modality.captures[1])

    # Preparing the observation likelihood array
    ol_array = Array{Float64}(undef, n_observations[modality], n_states...)

    for i in 1:n_observations[modality]
  
        # Makes sure that the dimensions always match no matter the number of states and factors
        inds = (i,)
        for _ in 1:length(n_states)
            inds = (inds..., :)
        end
    
        ol_array[inds...] .= reshape(parent_predictions[i], n_states...)
    end

    ol_array_normalized = normalize_arrays([ol_array])[1]

    return ol_array_normalized

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
function update_node_posterior!(node::OLStateNode, update_type::EnhancedUpdate)

    #Update posterior 
    node.states.posterior, node.states.posterior_observation = calculate_posterior(node)

    return nothing
end


function calculate_posterior(node::OLStateNode)

    # Extract the pomdp observation child
    child = node.edges.pomdp_observation_children[1]

    # Extract the modality of this node from name
    modality = match(r"m(\d+)", node.name)
    modality = parse(Int, modality.captures[1])

    # Extracting the correct posterior observations likelihood array (ola) based on the modality
    ola = child.states.posterior[modality]

    # Observation in this modality extracted from the ola
    observation = unique([I[1] for I in findall(x -> x != 0, ola)])[1]

    # Extracting the number of observations from the ola
    n_observations = size(ola)[1]

    # Creating the missing-onehot vector for passing onto the categorical parents
    one_hot_observation = Vector{Union{Missing, Int}}(missing, n_observations)
    one_hot_observation[observation] = 1

    # Defining the posterior and posterior observation
    posterior = ola
    posterior_observation = one_hot_observation
    
    return posterior, posterior_observation
end


function update_node_value_prediction_error!(node::OLStateNode)
    return nothing
end

function update_node_precision_prediction_error!(node::OLStateNode)
    return nothing
end



