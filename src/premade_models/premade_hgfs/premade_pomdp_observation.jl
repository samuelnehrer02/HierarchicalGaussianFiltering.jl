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

    ### Input node parent is the POMDP observation state node
    pomdp_input_parents_names = Vector{String}()

    ### POMDP observation state node parents are observation likelihood (ol) nodes
    pomdp_observation_state_parents_names = Vector{String}()

    ### POMDP ol parents are categorical nodes
    pomdp_ol_parents_names = Vector{String}()

    ### categorical node parents are binary nodes
    categorical_parents_names = Vector{String}()

    ### binary nodes parents are probability nodes
    binary_parents_names = Vector{String}()

    ### probability parent nodes are volatility node
    probability_parents_name = Vector{String}()

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

    ### Creating the pomdp observation state node name
    push!(pomdp_input_parents_names, "xpomdp_observation")

    ### Creating the observation likelihood nodes. One for each modality
    for i in 1:length(config["n_observations"])
        push!(pomdp_observation_state_parents_names, "xol_m$i")
    end

    ### Creating the categorical node names
    for i in 1:length(config["n_observations"])
        for j in 1:config["n_observations"][i]
            push!(pomdp_ol_parents_names, "xcat_m$(i)_o$(j)")
        end
    end
    
    # Creating the binary node names. Generates name for each unique combination of states
    for i in 1:length(config["n_observations"])
        for j in 1:config["n_observations"][i]
            for indices in Iterators.product((1:config["n_states"][k] for k in 1:length(config["n_states"]))...)

                name = "xbin_m$(i)_o$(j)" * join("_f$(k)s$(indices[k])" for k in 1:length(indices))
                push!(categorical_parents_names, name)

            end
        end
    end

    # Creating the probability node names
    for i in 1:length(config["n_observations"])
        for j in 1:config["n_observations"][i]
            for indices in Iterators.product((1:config["n_states"][k] for k in 1:length(config["n_states"]))...)

                name = "xprob_m$(i)_o$(j)" * join("_f$(k)s$(indices[k])" for k in 1:length(indices))
                push!(binary_parents_names, name)

            end
        end
    end

    ### Creating the top volatility node
    if config["include_volatility_parent"]
        push!(probability_parents_name, "xvol")
    end

    # Creating empty list to store the nodes
    nodes = Vector{AbstractNodeInfo}()

    ### For the POMDP input node
    push!(nodes, PomdpInput("upomdp"))

    ### For the POMDP state node
    push!(nodes, PomdpObservationState("xpomdp_observation"))

    ### For the OL nodes
    for node_name in pomdp_observation_state_parents_names
        push!(nodes, OLState(node_name))
    end

    ### For categorical nodes
    for node_name in pomdp_ol_parents_names
        push!(nodes, CategoricalState(node_name))
    end

    ### For binary nodes
    for node_name in categorical_parents_names
        push!(nodes, BinaryState(node_name))
    end

    ### For probability nodes
    for node_name in binary_parents_names
        push!(
            nodes, 
            ContinuousState(
                name = node_name,
                volatility = config[("xprob", "volatility")],
                drift = config[("xprob", "drift")],
                autoconnection_strength = config[("xprob", "autoconnection_strength")],
                initial_mean = config[("xprob", "initial_mean")],
                initial_precision = config[("xprob", "initial_precision")],
            ),
        )
        #Add the grouped parameter name to grouped parameters vector
        push!(grouped_parameters_xprob_initial_precision, (node_name, "initial_precision"))
        push!(grouped_parameters_xprob_initial_mean, (node_name, "initial_mean"))
        push!(grouped_parameters_xprob_volatility, (node_name, "volatility"))
        push!(grouped_parameters_xprob_drift, (node_name, "drift"))
        push!(
            grouped_parameters_xprob_autoconnection_strength,
            (node_name, "autoconnection_strength"),
        )
    end

    #If volatility parent is included
    if config["include_volatility_parent"]
        #Add the shared volatility parent of the continuous nodes
        push!(
            nodes,
            ContinuousState(
                name = "xvol",
                volatility = config[("xvol", "volatility")],
                drift = config[("xvol", "drift")],
                autoconnection_strength = config[("xvol", "autoconnection_strength")],
                initial_mean = config[("xvol", "initial_mean")],
                initial_precision = config[("xvol", "initial_precision")],
            ),
        )

    end

    # Creating the edges
    ## Empty list for storing the edges
    edges = OrderedDict{Tuple{String,String}, CouplingType}()

    ### Creating edges between input pomdp and observation state pomdp
    for (pomdp_input_node_name, pomdp_input_parents_name) in zip(pomdp_input_node_names, pomdp_input_parents_names)
        push!(edges, (pomdp_input_node_name, pomdp_input_parents_name) => ObservationCoupling())
    end

    ### Creating edges between observation state pomdp and observation likelihood (OL) nodes
    for pomdp_input_parents_name in pomdp_input_parents_names
        for pomdp_observation_state_parents_name in pomdp_observation_state_parents_names
            push!(edges, (pomdp_input_parents_name, pomdp_observation_state_parents_name) => PomdpObservationCoupling())
        end
    end

    ### Creating edges between OL and categorical nodes
    for i in 1:length(config["n_observations"])
        for k in 1:config["n_observations"][i]
    
            parent_name = filter(x -> occursin("m$(i)_o$(k)", x), pomdp_ol_parents_names)
            child_name = filter(x -> occursin("m$(i)", x), pomdp_observation_state_parents_names)
    
            push!(edges, (child_name[1], parent_name[1]) => OLCoupling())
    
        end
    end

    ## Creating edges between categorical and binary nodes
    for i in 1:length(config["n_observations"])
        for k in 1:config["n_observations"][i]

            parent_names = filter(x -> occursin("m$(i)_o$(k)", x), categorical_parents_names)
            child_names = filter(x -> occursin("m$(i)_o$(k)", x), pomdp_ol_parents_names)

            for parent_name in parent_names
                for child_name in child_names
                    push!(edges, (child_name, parent_name) => CategoryCoupling())
                end
            end

        end
    end

    for name in categorical_parents_names
        pattern = replace(name, "xbin_" => "")

        parent_name = filter(x -> occursin(pattern, x), binary_parents_names)
        child_name = name

        push!(edges, (child_name, parent_name[1]) => ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")]))

        push!(
            grouped_parameters_xbin_xprob_coupling_strength,
            (child_name[1], parent_name[1], "coupling_strength"),
        )

    end

    ### Creating edges between probability nodes and volatility node
    if config["include_volatility_parent"]
        for name in binary_parents_names
            parent_name = "xvol"
            child_name = name

            push!(edges, (child_name, parent_name) => VolatilityCoupling(config[("xprob", "xvol", "coupling_strength")]))

            push!(
                grouped_parameters_xprob_xvol_coupling_strength,
                (child_name, "xvol", "coupling_strength"),
            )

        end
    end

    # Create dictionary with shared parameter information
    parameter_groups = [
        ParameterGroup(
            "xprob_volatility",
            grouped_parameters_xprob_volatility,
            config[("xprob", "volatility")],
        ),
        ParameterGroup(
            "xprob_initial_precision",
            grouped_parameters_xprob_initial_precision,
            config[("xprob", "initial_precision")],
        ),
        ParameterGroup(
            "xprob_initial_mean",
            grouped_parameters_xprob_initial_mean,
            config[("xprob", "initial_mean")],
        ),
        ParameterGroup(
            "xprob_drift",
            grouped_parameters_xprob_drift,
            config[("xprob", "drift")],
        ),
        ParameterGroup(
            "xprob_autoconnection_strength",
            grouped_parameters_xprob_autoconnection_strength,
            config[("xprob", "autoconnection_strength")],
        ),
        ParameterGroup(
            "xbin_xprob_coupling_strength",
            grouped_parameters_xbin_xprob_coupling_strength,
            config[("xbin", "xprob", "coupling_strength")],
        ),
    ]

    # If volatility parent is included
    if config["include_volatility_parent"]
        push!(
            parameter_groups,
            ParameterGroup(
                "xprob_xvol_coupling_strength",
                grouped_parameters_xprob_xvol_coupling_strength,
                config[("xprob", "xvol", "coupling_strength")],
            ),
        )
    end

    ### Initializing the HGF
    hgf = init_hgf(
        nodes = nodes,
        edges = edges,
        parameter_groups = parameter_groups,
        verbose = false,
        node_defaults = NodeDefaults(update_type = config["update_type"]),
        save_history = config["save_history"]
    )

    hgf.state_nodes["xpomdp_observation"].states.n_states = config["n_states"]
    hgf.state_nodes["xpomdp_observation"].states.n_observations = config["n_observations"]
   

    return hgf

end
