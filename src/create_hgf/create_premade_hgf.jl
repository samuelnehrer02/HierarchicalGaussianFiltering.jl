"""
    premade_hgf(model_name::String, config::Dict = Dict(); verbose = true)

Create an HGF from the list of premade HGFs.

# Arguments
 - 'model_name::String': Name of the premade HGF. Returns a list of possible model names if set to 'help'. 
 - 'config::Dict = Dict()': A dictionary with configurations for the HGF, like parameters and settings.
 - 'verbose::Bool = true': If set to false, warnings are hidden.
"""
function premade_hgf(model_name::String, config::Dict = Dict(); verbose = true)

    #A list of all the included premade models
    premade_models = Dict(
        "continuous_2level" => premade_continuous_2level,   #The standard continuous input 2 level HGF
        "binary_2level" => premade_binary_2level,           #The standard binary input 2 level HGF
        "binary_3level" => premade_binary_3level,           #The standard binary input 3 level HGF
        "JGET" => premade_JGET,                             #The JGET model
        "categorical_3level" => premade_categorical_3level, #The standard categorical input 3 level HGF
        "categorical_state_transitions" => premade_categorical_state_transitions, #Categorical 3 level HGF for learning state transitions
        "pomdp_transitions" => premade_pomdp_transition, #3 level HGF for POMDP transitions 
        "pomdp_transitions_2level" => premade_pomdp_transition_2level, # POMDP transitions without volatility node
    )

    #Check that the specified model is in the list of keys
    if model_name in keys(premade_models)
        #Create the specified model
        return premade_models[model_name](config, verbose = verbose)
        #If the user asked for help
    elseif model_name == "help"
        #Return the list of keys
        print(keys(premade_models))
        return nothing
        #If an invalid name is given
    else
        #Raise an error
        throw(
            ArgumentError(
                "the specified string does not match any model. Type premade_hgf('help') to see a list of valid input strings",
            ),
        )
    end
end
