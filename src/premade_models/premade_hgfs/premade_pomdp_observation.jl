"""
    premade_pomdp_observation_3level(config::Dict; verbose::Bool = true)

The pomdp observation 3 level HGF model, learns state observation to state probabilities.
This HGF has five shared parameters: 
"xprob_volatility"
"xprob_initial_precisions"
"xprob_initial_means"
"coupling_strengths_xbin_xprob"
"coupling_strengths_xprob_xvol"

# Config defaults:
    - "n_categories": 4
    - ("xprob", "volatility"): -2
    - ("xvol", "volatility"): -2
    - ("xbin", "xprob", "coupling_strength"): 1
    - ("xprob", "xvol", "coupling_strength"): 1
    - ("xprob", "initial_mean"): 0
    - ("xprob", "initial_precision"): 1
    - ("xvol", "initial_mean"): 0
    - ("xvol", "initial_precision"): 1
"""

function premade_pomdp_observation(config::Dict; verbose::Bool = true)

    # Initial configurations for the Total POMDP HGF model
    defaults = Dict(
        "n_states" => [4, 2],
        "n_observations" => [4, 3, 2],
        "include_volatility_parent" => true,
        ("xprob", "volatility") => -2,
        ("xprob", "drift") => 0,
        ("xprob", "autoconnection_strength") => 1,
        ("xprob", "initial_mean") => 0,
        ("xprob", "initial_precision") => 1,
        ("xvol", "volatility") => -2,
        ("xvol", "drift") => 0,
        ("xvol", "autoconnection_strength") => 1,
        ("xvol", "initial_mean") => 0,
        ("xvol", "initial_precision") => 1,
        ("xbin", "xprob", "coupling_strength") => 1,
        ("xprob", "xvol", "coupling_strength") => 1,
        "update_type" => EnhancedUpdate(),
        "save_history" => true,
    )

    if verbose
        warn_premade_defaults(defaults, config)
    end

    # Merge to overwrite defaults
    config = merge(defaults, config)

    # Preparing names for input nodes (one POMDP input node)
    pomdp_input_node_names = Vector{String}()

    ### Observation parent is the POMDP state node
    pomdp_input_parents_names = Vector{String}()

    ### POMDP parents are observation likelihood (ol) nodes
    pomdp_ol_parents_names = Vector{String}()

    ### pomdp_ol parents are categorical nodes 
    tpm_parents_names = Vector{String}()

    ### Categorical parents are binary nodes
    category_parents_names = Vector{String}()

    ### Binary parents are continuous nodes
    binary_parents_names = Vector{String}()

    ### Continuous parents are continuous volatility nodes
    continuous_parents_names = Vector{String}()

    ### Empty lists for grouped parameters
    grouped_parameters_xprob_initial_precision = []
    grouped_parameters_xprob_initial_mean = []
    grouped_parameters_xprob_volatility = []
    grouped_parameters_xprob_drift = []
    grouped_parameters_xprob_autoconnection_strength = []
    grouped_parameters_xbin_xprob_coupling_strength = []
    grouped_parameters_xprob_xvol_coupling_strength = []

    ### Creating the pomdp input node
    push!(pomdp_input_node_names, "upomdp")

    ### Creating the pomdp state node
    for i in 1:length(config["n_observations"])
        push!(pomdp_input_parents_names, "xpomdp_ol_m$i")
    end
    push!(pomdp_input_parents_names, "xpomdp_ol")
    






end


config = Dict(
    "n_states" => [4, 2],
    "n_observations" => [4, 3, 2],
    "include_volatility_parent" => true,
    ("xprob", "volatility") => -2,
    ("xprob", "drift") => 0,
    ("xprob", "autoconnection_strength") => 1,
    ("xprob", "initial_mean") => 0,
    ("xprob", "initial_precision") => 1,
    ("xvol", "volatility") => -2,
    ("xvol", "drift") => 0,
    ("xvol", "autoconnection_strength") => 1,
    ("xvol", "initial_mean") => 0,
    ("xvol", "initial_precision") => 1,
    ("xbin", "xprob", "coupling_strength") => 1,
    ("xprob", "xvol", "coupling_strength") => 1,
    "update_type" => EnhancedUpdate(),
    "save_history" => true,
)

pomdp_input_parents_names = Vector{String}()

for i in 1:length(config["n_observations"])
    push!(pomdp_input_parents_names, "xpomdp_ol_m" * string(i))
end

pomdp_input_parents_names











