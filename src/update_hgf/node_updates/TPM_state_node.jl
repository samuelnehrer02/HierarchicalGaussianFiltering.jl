








function calculate_posterior(node::TPMStateNode)
    # Extract the pomdp child
    child = node.edges.pomdp_children[1]

    # Initialize posterior as previous posterior
    posterior = node.states.posterior

    # If Categorical State Node has POMDP children we just copy the input
    else
        # Extract the pomdp child
        child = node.edges.pomdp_children[1]

        # Initialize posterior as previous posterior
        posterior = node.states.posterior

        # Set the posterior to be the input value
        posterior .= child.states.input_value
    end

    return posterior
end


