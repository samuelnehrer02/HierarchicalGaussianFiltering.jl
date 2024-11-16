"""
    premade_pomdp_transition_2level(config::Dict; verbose::Bool = true)

The pomdp transition 2 level HGF model, learns state transition probabilities between a set of n categorical states.
This HGF has five shared parameters: 
"xprob_initial_precisions"
"xprob_initial_means"
"coupling_strengths_xbin_xprob"

# Config defaults:
    - "n_categories": 4
    - ("xbin", "xprob", "coupling_strength"): 1
    - ("xprob", "initial_mean"): 0
    - ("xprob", "initial_precision"): 1
"""
function premade_pomdp_transition_2level(config::Dict; verbose::Bool = true)

    # Initial configurations for the Total POMDP HGF model
    defaults = Dict(
        "n_states" => nothing,
        "n_categories_from" => [4, 2],
        "n_categories_to" => [4, 2],
        "n_control" => [4, 1],
        "include_volatility_parent" => false,
        ("xprob", "drift") => 0,
        ("xprob", "autoconnection_strength") => 1,
        ("xprob", "initial_mean") => 0,
        ("xprob", "initial_precision") => 1,
        ("xbin", "xprob", "coupling_strength") => 1,
        "update_type" => EnhancedUpdate(),
        "save_history" => true,
    )

    if verbose
        warn_premade_defaults(defaults, config)
    end

    #Merge to overwrite defaults
    config = merge(defaults, config)

    if config["n_states"] != nothing
        config["n_categories_from"] = config["n_states"]
        config["n_categories_to"] = config["n_states"]
    end

    # Preparing names for input nodes (one POMDP input node)
    pomdp_input_node_names = Vector{String}()

    ### Observation parent is the POMDP state node
    pomdp_input_parents_names = Vector{String}()

    ### POMDP parents are TPM nodes 
    pomdp_state_parents_names = Vector{String}()

    ### TMB parents are categorical nodes 
    tpm_parents_names = Vector{String}()

    ### Categorical parents are binary nodes
    category_parents_names = Vector{String}()

    ### Binary parents are continuous nodes
    binary_parents_names = Vector{String}()

    ### Empty lists for grouped parameters
    grouped_parameters_xprob_initial_precision = []
    grouped_parameters_xprob_initial_mean = []
    grouped_parameters_xprob_drift = []
    grouped_parameters_xprob_autoconnection_strength = []
    grouped_parameters_xbin_xprob_coupling_strength = []

    ### Creating the pomdp input node
    push!(pomdp_input_node_names, "upomdp")

    ### Creating the pomdp state node
    push!(pomdp_input_parents_names, "xpomdp")

    ### Creating proper node names
    for j in 1:length(config["n_control"])

        ### TPM nodes per action
        for i in 1:config["n_control"][j]
            push!(pomdp_state_parents_names, "xtpm_f" * string(j) *"_a" *string(i))
        end

        ### Category nodes per state_from and action tpm node
        for k in 1:config["n_control"][j]
            for i in 1:config["n_categories_from"][j]
                push!(tpm_parents_names, "xcat_f" * string(j) * "_a" * string(k) * "_" * string(i))
            end
        end

        ### Binary nodes per action per state_from_to
        for x in 1:config["n_control"][j] 
            for i in 1:config["n_categories_from"][j]
                for k in 1:config["n_categories_to"][j]
                    push!(category_parents_names, "xbin_f" * string(j) * "_a" * string(x) * "_" * string(i) * "_" * string(k))
                end
            end
        end

        ### Continuous nodes per state_from_to
        for x in 1:config["n_control"][j]
            for i in 1:config["n_categories_from"][j]
                for k in 1:config["n_categories_to"][j]
                    push!(binary_parents_names, "xcon_f" * string(j) * "_a" * string(x) * "_" * string(i) * "_" * string(k))
                end
            end
        end
    end


    # Creating empty list to store the nodes
    nodes = Vector{AbstractNodeInfo}()

    ### For the POMDP input node
    push!(nodes, PomdpInput("upomdp"))

    ### For the POMDP state node
    push!(nodes, PomdpState("xpomdp"))

    ### For the TPM nodes
    for node_name in pomdp_state_parents_names
        push!(nodes, TPMState(node_name))
    end

    ### For categorical nodes
    for node_name in tpm_parents_names
        push!(nodes, CategoricalState(node_name))
    end

    ### For binary nodes
    for node_name in category_parents_names
        push!(nodes, BinaryState(node_name))
    end

    ### For continuous nodes
    for node_name in binary_parents_names
        push!(nodes, 
        ContinuousState(
            name = node_name,
            drift = config[("xprob", "drift")],
            autoconnection_strength = config[("xprob", "autoconnection_strength")],
            initial_mean = config[("xprob", "initial_mean")],
            initial_precision = config[("xprob", "initial_precision")],
            )
        )
        #Add the grouped parameter name to grouped parameters vector
        push!(grouped_parameters_xprob_initial_precision, (node_name, "initial_precision"))
        push!(grouped_parameters_xprob_initial_mean, (node_name, "initial_mean"))
        push!(grouped_parameters_xprob_drift, (node_name, "drift"))
        push!(
            grouped_parameters_xprob_autoconnection_strength,
            (node_name, "autoconnection_strength"),
        )
    end


    # Creating the edges
    ## Empty list for storing the edges
    edges = OrderedDict{Tuple{String,String},CouplingType}()

    ### Creating edges between input pomdp and state pomdp
    for (pomdp_input_node_name, pomdp_input_parents_name) in zip(pomdp_input_node_names, pomdp_input_parents_names)
        push!(edges, (pomdp_input_node_name, pomdp_input_parents_name) => PomdpCoupling())
    end

    ### Creating edges between state pomdp and tpm
    for pomdp_input_parents_name in pomdp_input_parents_names
        for pomdp_state_parents_name in pomdp_state_parents_names
            push!(edges, (pomdp_input_parents_name, pomdp_state_parents_name) => PomdpCoupling())
        end
    end

    ### Creating edges between tpm and categorical
    for j in 1:length(config["n_control"])
        for i in 1:config["n_control"][j]
            for k in 1:config["n_categories_from"][j]

                parent_name = filter(x -> occursin("f$(j)_a$(i)_$(k)", x), tpm_parents_names)
                child_name = filter(x -> occursin("f$(j)_a$(i)", x), pomdp_state_parents_names)

                push!(edges, (child_name[1], parent_name[1]) => TPMCoupling())

            end
        end
    end


    ### Creating edges between categorical and binary nodes
    for j in 1:length(config["n_control"])
        for i in 1:config["n_control"][j]
            for k in 1:config["n_categories_from"][j]
                for m in 1:config["n_categories_to"][j]

                    parent_name = filter(x -> occursin("f$(j)_a$(i)_$(k)_$(m)", x), category_parents_names)
                    child_name = filter(x -> occursin("f$(j)_a$(i)_$(k)", x), tpm_parents_names)
                    
                    push!(edges, (child_name[1], parent_name[1]) => CategoryCoupling())

                end
            end
        end
    end

    ### Creating edges between binary and continuous nodes
    for j in 1:length(config["n_control"])
        for i in 1:config["n_control"][j]
            for k in 1:config["n_categories_from"][j]
                for m in 1:config["n_categories_to"][j]

                    parent_name = filter(x -> occursin("f$(j)_a$(i)_$(k)_$(m)", x), binary_parents_names)
                    child_name = filter(x -> occursin("f$(j)_a$(i)_$(k)_$(m)", x), category_parents_names)
                    
                    push!(edges, (child_name[1], parent_name[1]) => ProbabilityCoupling(config[("xbin", "xprob", "coupling_strength")]))

                end
            end
        end
    end

    # Create dictionary with shared parameter information
    parameter_groups = [
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


    ### Initializing the HGF
    hgf = init_hgf(
        nodes = nodes,
        edges = edges,
        parameter_groups = parameter_groups,
        verbose = false,
        node_defaults = NodeDefaults(update_type = config["update_type"]),
        save_history = config["save_history"]
    )

    hgf.state_nodes["xpomdp"].states.n_control = config["n_control"]
    hgf.state_nodes["xpomdp"].states.n_states = config["n_states"]

    reset_state!(hgf.state_nodes["xpomdp"])

    hgf.state_nodes["xpomdp"].states.n_control = config["n_control"]
    hgf.state_nodes["xpomdp"].states.n_states = config["n_states"]

    return hgf
end

# config = Dict(
#     "n_categories" => [10, 2],
#     "n_control" => [4, 1],
# )


# premade_pomdp_transition_3level(config, true)

