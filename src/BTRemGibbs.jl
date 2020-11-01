"""
BTR_EMGibbs plus interaction effects between x variables and topics in the regression
    order is (all topics)*x_1 + (all_topics)*x_2 etc...
"""
function BTRemGibbs(btrmodel::BTRModel)

    opts = btrmodel.options

    ## Check that everything matches up
    @assert opts.ntopics == btrmodel.crps.ntopics "ntopics in corpus and options must match"
    @assert all(in.(opts.interactions, [opts.xregs])) "Can't have interactions that aren't specified in xreg"
    @assert btrmodel.crps.V == length(btrmodel.vocab) "Vocab length in BTRModel and V in BTRCorpus do not match"
    @assert length(btrmodel.ω) == (opts.ntopics + length(opts.interactions)*opts.ntopics +
        (length(opts.xregs) - length(opts.interactions))) "Dimensions of ω do not match with xregs and interactions"

    # Make necessary changes if using Cross-Validation EM
    if opts.CVEM == :obs
        E_btrmodel, M_btrmodel, E_obs, M_obs = btrmodel_splitobs(btrmodel, opts.CVEM_split)
    else
        E_btrmodel = btrmodel
        M_btrmodel = btrmodel
        E_obs = 1:btrmodel.crps.N
        M_obs = 1:btrmodel.crps.N
    end

    for em in 1:opts.EM_iters
        ω_iters = btrmodel.ω_iters::Array{Float64,2}
        display(join(["E step ", em]))
        E_btrmodel = BTREstep(E_btrmodel)
        heatmap(E_btrmodel.β)
        M_btrmodel.β = E_btrmodel.β

        display(join(["M step ", em]))
        if opts.CVEM == :obs
            M_btrpredict = BTRpredict(M_btrmodel.crps, E_btrmodel)
            M_btrmodel.Z_bar = Matrix(transpose(M_btrpredict.Z_bar))
            M_btrmodel.regressors = M_btrpredict.regressors
            M_btrmodel.crps = M_btrpredict.crps
        end
        M_btrmodel = BTRMstep(M_btrmodel)
        display(join(["EM iteration ", em, " complete, MSE: ", M_btrmodel.σ2]))
        E_btrmodel.ω = M_btrmodel.ω
        E_btrmodel.σ2 = M_btrmodel.σ2

        # Find coefficient updates
        ω_iters[:,em+1]= M_btrmodel.ω
        ω_diff = abs.(ω_iters[:,em+1] .- ω_iters[:,em])
        ω_diff[ω_diff.<opts.ω_tol] = zeros(sum(ω_diff.<opts.ω_tol))
        if opts.rel_tol
            ω_diff ./=abs.(M_btrmodel.ω)
        end

        # Plot update on ω if specified
        if opts.plot_ω
            display(plot(0:em,transpose(ω_iters[:,1:(em+1)]),legend=false,
            xticks = 0:em, xlab = "EM iteration", ylab = "Omega updates"))
            display(join(["Coefficient updates:", ω_diff], " "))
        end

        if maximum(ω_diff) < opts.ω_tol
            break
        end
    end

    if opts.CVEM == :obs
        btrmodel = btrmodel_combineobs(btrmodel, E_btrmodel, M_btrmodel,E_obs, M_obs)
    end

    return btrmodel
end



"""
Split BTRModel for CVEM approach
"""
function btrmodel_splitobs(btrmodel::BTRModel, split_ratio::Float64)
    # Randomly split into E and M observations
    idx = shuffle(1:btrmodel.crps.N)
    # Separate indices into test and training
    E_obs = Array{Int64,1}(view(idx, 1:floor(Int, opts.CVEM_split*btrmodel.crps.N)))::Array{Int64,1}
    M_obs = setdiff(idx, E_obs)::Array{Int64,1}

    ## Extract btrmodel for E-step
    E_btrmodel = deepcopy(btrmodel)
    # Update corpus
    E_crps = E_btrmodel.crps
    E_crps.docs = E_crps.docs[E_obs]
    E_crps.topics = gettopics(E_crps.docs)
    E_crps.docidx_labels = E_crps.docidx_labels[E_obs]
    E_crps.N = length(E_crps.docs)
    # Other elements
    E_btrmodel.y = E_btrmodel.y[E_obs]
    E_btrmodel.regressors = E_btrmodel.regressors[E_obs,:]
    E_btrmodel.Z_bar = E_btrmodel.Z_bar[:,E_obs]

    ## Extract btrmodel for M-step
    M_btrmodel = deepcopy(btrmodel)
    # Update corpus
    M_crps = M_btrmodel.crps
    M_crps.docs = M_crps.docs[M_obs]
    M_crps.topics = gettopics(M_crps.docs)
    M_crps.docidx_labels = M_crps.docidx_labels[M_obs]
    M_crps.N = length(E_crps.docs)
    # Other elements
    M_btrmodel.y = M_btrmodel.y[M_obs]
    M_btrmodel.regressors = M_btrmodel.regressors[M_obs,:]
    M_btrmodel.Z_bar = M_btrmodel.Z_bar[:,M_obs]

    return E_btrmodel,M_btrmodel,E_obs,M_obs
end


function btrmodel_combineobs(btrmodel::BTRModel, E_btrmodel::BTRModel, M_btrmodel::BTRModel,
        E_obs::Array{Int64,1}, M_obs::Array{Int64,1})
    # Randomly split into E and M observations
    N = E_btrmodel.crps.N + M_btrmodel.crps.N
    @assert N == maximum(vcat(E_obs, M_obs)) "E and M step subsample sizes don't match total data"
    @assert N == btrmodel.crps.N "E and M step subsample sizes don't match total data"
    @assert E_btrmodel.β == M_btrmodel.β "E and M step β don't match"
    @assert E_btrmodel.ω == M_btrmodel.ω "E and M step β don't match"

    # Update regression parameters
    btrmodel.ω = M_btrmodel.ω
    btrmodel.ω_post = M_btrmodel.ω_post
    btrmodel.σ2 = M_btrmodel.σ2
    btrmodel.σ2_post = M_btrmodel.σ2_post

    # Update corpus
    btrmodel.crps.docs[E_obs] = E_btrmodel.crps.docs
    btrmodel.crps.docs[M_obs] = M_btrmodel.crps.docs
    btrmodel.crps.topics = gettopics(btrmodel.crps.docs)
    btrmodel.β = E_btrmodel.β

    # Other elements
    btrmodel.Z_bar[:,E_obs] = E_btrmodel.Z_bar
    btrmodel.Z_bar[:,M_obs] = M_btrmodel.Z_bar
    btrmodel.regressors[E_obs,:] = E_btrmodel.regressors
    btrmodel.regressors[M_obs,:] = M_btrmodel.regressors

    return btrmodel
end
