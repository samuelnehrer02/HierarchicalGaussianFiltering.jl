###################################
######## POMDP input node #########
###################################

##### Superfunction #####
"""
    update_node_prediction!(node::PomdpInputNode)

There is no prediction update for POMDP input nodes, as the prediction precision is constant.
"""
function update_node_prediction!(node::PomdpInputNode, stepsize::Real)
    return nothing
end


###############################################
######## Update value prediction error ########
###############################################

##### Superfunction #####
"""
    update_node_value_prediction_error!(node::PomdpInputNode)

    There is no value prediction error update for POMDP input nodes.
"""
function update_node_value_prediction_error!(node::PomdpInputNode)
    return nothing
end


###################################################
######## Update precision prediction error ########
###################################################

##### Superfunction #####
"""
    update_node_precision_prediction_error!(node::PomdpInputNode)

There is no volatility prediction error update for POMDP input nodes.
"""
function update_node_precision_prediction_error!(node::PomdpInputNode)
    return nothing
end
