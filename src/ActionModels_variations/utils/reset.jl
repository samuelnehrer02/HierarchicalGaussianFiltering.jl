"""
    reset!(hgf::HGF)

Reset an HGF to its initial state.
"""
function ActionModels.reset!(hgf::HGF)

    #Reset the timesteps for the HGF
    hgf.timesteps = [0]

    #Go through each node
    for node in hgf.ordered_nodes.all_nodes

        #Reset its state
        reset_state!(node)

        #For each state in the history
        for state_name in fieldnames(typeof(node.history))

            #Empty the history
            empty!(getfield(node.history, state_name))

            #Add the new current state as the first state in the history
            push!(getfield(node.history, state_name), getfield(node.states, state_name))
        end
    end
end


function reset_state!(node::ContinuousStateNode)

    node.states.posterior_mean = node.parameters.initial_mean
    node.states.posterior_precision = node.parameters.initial_precision

    node.states.value_prediction_error = missing
    node.states.precision_prediction_error = missing

    node.states.prediction_mean = missing
    node.states.prediction_precision = missing
    node.states.effective_prediction_precision = missing

    return nothing
end

function reset_state!(node::ContinuousInputNode)

    node.states.input_value = missing

    node.states.value_prediction_error = missing
    node.states.precision_prediction_error = missing

    node.states.prediction_mean = missing
    node.states.prediction_precision = missing

    return nothing
end

function reset_state!(node::BinaryStateNode)

    node.states.posterior_mean = missing
    node.states.posterior_precision = missing

    node.states.value_prediction_error = missing

    node.states.prediction_mean = missing
    node.states.prediction_precision = missing

    return nothing
end

function reset_state!(node::BinaryInputNode)

    node.states.input_value = missing

    return nothing
end

function reset_state!(node::CategoricalStateNode)

    node.states.posterior .= missing
    node.states.value_prediction_error .= missing
    node.states.prediction .= 1/length(node.states.prediction)
    node.states.parent_predictions .= 1/length(node.states.parent_predictions)

    return nothing
end

function reset_state!(node::CategoricalInputNode)

    node.states.input_value = [missing]

    return nothing
end

function reset_state!(node::PomdpInputNode)

    node.states.input_value = missing
    node.states.policy_chosen = missing

    return nothing
end

function reset_state!(node::PomdpStateNode)

    # Set the posterior to an empty matrix
    node.states.posterior = Vector{Vector{<:Real}}(undef, 0)
    
    if !ismissing(node.states.n_states[1])
        node.states.previous_qs = [fill(1 / n, n) for n in node.states.n_states]
    else
        node.states.previous_qs = Vector{Union{Real, Missing}}(undef, 0)
    end

    # node.states.prediction = Array{Matrix{Union{Real, Missing}}, 1}(undef, 0)
    node.states.prediction = missing
    node.states.parent_predictions = Matrix{Union{Real, Missing}}(undef, 0, 0)

    node.states.posterior_policy = Vector{Vector{Union{Missing, Int64}}}(undef, 0)
    node.states.n_control = [missing]
    node.states.n_states = [missing]

    return nothing
end


function reset_state!(node::TPMStateNode)

    # Check how many categories the node has
    n_categories = length(node.edges.tpm_parents)

    # Set the posterior to an empty matrix
    node.states.posterior = Matrix{Union{Real, Missing}}(missing, n_categories, n_categories)
    
    node.states.previous_qs = Vector{Union{Real, Missing}}(missing, n_categories)
    node.states.previous_qs .= 1/n_categories

    node.states.prediction .= 1/n_categories
    node.states.parent_predictions .= 1/n_categories

    return nothing
end



