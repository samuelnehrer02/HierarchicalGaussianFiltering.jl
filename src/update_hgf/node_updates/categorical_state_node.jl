###################################
######## Update prediction ########
###################################

##### Superfunction #####
"""
    update_node_prediction!(node::CategoricalStateNode)

Update the prediction of a single categorical state node.
"""
function update_node_prediction!(node::CategoricalStateNode, stepsize::Real)

    #Update prediction mean
    node.states.prediction, node.states.parent_predictions = calculate_prediction(node)
    
    return nothing
end


@doc raw"""
    function calculate_prediction(node::CategoricalStateNode)

Calculate the prediction for a categorical state node.

Uses the equation
``  \vec{\hat{\mu}_n}= \frac{\hat{\mu}_j}{\sum_{j=1}^{j\;binary \;parents} \hat{\mu}_j} ``
"""
function calculate_prediction(node::CategoricalStateNode)

    #Get parent posteriors
    parent_posteriors =
        map(x -> x.states.posterior_mean, collect(values(node.edges.category_parents)))

    #Get current parent predictions
    parent_predictions =
        map(x -> x.states.prediction_mean, collect(values(node.edges.category_parents)))

    #Get previous parent predictions
    previous_parent_predictions = node.states.parent_predictions

    #If there was an observation
    if any(!ismissing, node.states.posterior)

        #Calculate implied learning rate
        implied_learning_rate =
            (
                (parent_posteriors .- previous_parent_predictions) ./
                ((parent_predictions .- previous_parent_predictions) .+ 1e-16)
            ) .- 1
        
        #Calculate the prediction mean
        prediction =
            ((implied_learning_rate .* parent_predictions) .+ 1) ./
            sum(implied_learning_rate .* parent_predictions .+ 1)
        
        #If there was no observation
    else
        #Extract prediction from last timestep
        prediction = node.states.prediction

    end

    return prediction, parent_predictions

end


##################################
######## Update posterior ########
##################################

##### Superfunction #####
"""
    update_node_posterior!(node::CategoricalStateNode)

Update the posterior of a single categorical state node.
"""
function update_node_posterior!(node::CategoricalStateNode, update_type::ClassicUpdate)

    #Update posterior mean
    node.states.posterior = calculate_posterior(node)

    return nothing
end


@doc raw"""
    calculate_posterior(node::CategoricalStateNode)

Calculate the posterior for a categorical state node.

One hot encoding
`` \vec{u} = [0, 0, \dots ,1, \dots,0]  ``
"""
# For categorical state node with TPM children
function calculate_posterior(node::CategoricalStateNode)

    # If Categorical State Node has observation children
    if !isempty(node.edges.observation_children)
        #Extract the observation child
        child = node.edges.observation_children[1]

        #Initialize posterior as previous posterior
        posterior = node.states.posterior

        #For missing inputs
        if ismissing(child.states.input_value)
            #Set the posterior to be all missing
            posterior .= missing

        else
            #Set all values to 0
            posterior .= zero(Real)

            #Set the posterior for the observed category to 1
            posterior[child.states.input_value] = 1
        end

    elseif !isempty(node.edges.tpm_children)

        # Extract the TPM child
        child = node.edges.tpm_children[1]

        # Extract child posterior
        posterior_child = child.states.posterior

        if !ismissing(posterior_child)

            # Extracts the number from the name string
            cat_state = parse(Int, match(r"_(\d+)", node.name).captures[1])

            # Extracts the right row from the TPM child posterior matrix
            posterior = child.states.posterior[cat_state, :]

        else
            posterior = fill(missing, length(node.edges.category_parents))
        end
    

    elseif !isempty(node.edges.ol_children)

        # Extract the OL child
        child = node.edges.ol_children[1]

        # Extract the observation for this node based on its name
        node_name = node.name

        obs_node = match(r"o(\d+)", node_name)
        obs_node = parse(Int, obs_node.captures[1])

        # Extracting the observation posterior from child node_updates
        observation_vector = child.states.posterior_observation

        # If observation is equal to the obs_node (meaning the observation is for this node) then update the categorical node
        if !ismissing(observation_vector[obs_node])

            # Extracting the posterior from the child
            ola = child.states.posterior

            # Indexing the observation in question
            correct_obs_rows = ola[obs_node, :, ntuple(_ -> :, ndims(ola) - 2)...]

            # Making it into a vector to pass onto the binary parent nodes
            posterior = vec(correct_obs_rows)

        else
            # Else fill with a vector of missing, the length of the number of parents
            posterior = fill(missing, length(node.edges.category_parents))
        end

    end

    return posterior
end


###############################################
######## Update value prediction error ########
###############################################

##### Superfunction #####
"""
    update_node_value_prediction_error!(node::AbstractStateNode)

Update the value prediction error of a single state node.
"""
function update_node_value_prediction_error!(node::CategoricalStateNode)
    #Update value prediction error
    node.states.value_prediction_error = calculate_value_prediction_error(node)

    return nothing
end

@doc raw"""
    calculate_value_prediction_error(node::CategoricalStateNode)

Calculate the value prediction error for a categorical state node.

Uses the equation
`` \delta_n= u - \sum_{j=1}^{j\;value\;parents} \hat{\mu}_{j}  ``
"""
function calculate_value_prediction_error(node::CategoricalStateNode)

    #Get the prediction error for each category
    value_prediction_error = node.states.posterior - node.states.prediction

    return value_prediction_error
end


###################################################
######## Update precision prediction error ########
###################################################

##### Superfunction #####
"""
    update_node_precision_prediction_error!(node::CategoricalStateNode)

There is no volatility prediction error update for categorical state nodes.
"""
function update_node_precision_prediction_error!(node::CategoricalStateNode)
    return nothing
end
