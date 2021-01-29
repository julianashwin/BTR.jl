"""
BTR_EMGibbs plus interaction effects between x variables and topics in the regression
    order is (all topics)*x_1 + (all_topics)*x_2 etc...
"""
function BTRemGibbs(btrmodel::BTRModel)

    opts = btrmodel.options
    btrmodel.converged = false # Set convergence to false at the beginning

    ## Check that everything matches up
    @assert opts.ntopics == btrmodel.crps.ntopics "ntopics in corpus and options must match"
    @assert all(in.(opts.interactions, [opts.xregs])) "Can't have interactions that aren't specified in xreg"
    @assert btrmodel.crps.V == length(btrmodel.crps.vocab) "Vocab length in BTRModel and V in BTRCorpus do not match"
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
    # Initialise MSE with unconditional variance of y
    btrmodel.mse_iters[1] = var(M_btrmodel.y)

    # Begin EM loop
    for em in 1:opts.EM_iters
        ω_iters = btrmodel.ω_iters::Array{Float64,2}
        mse_iters = btrmodel.mse_iters::Array{Float64,1}
        mse_conv = opts.mse_conv::Int64
        display(join(["E step ", em]))
        E_btrmodel = BTREstep(E_btrmodel)
        #heatmap(E_btrmodel.β)
        M_btrmodel.β = E_btrmodel.β

        display(join(["M step ", em]))
        if opts.CVEM == :obs
            M_btrpredict = BTRpredict(M_btrmodel.crps, E_btrmodel)
            M_btrmodel.Z_bar = Matrix(transpose(M_btrpredict.Z_bar))
            M_btrmodel.regressors = M_btrpredict.regressors
            M_btrmodel.crps = M_btrpredict.crps
        end
        M_btrmodel = BTRMstep(M_btrmodel)
        # Add new mse to mse_iters
        mse_new = mean((M_btrmodel.y - M_btrmodel.regressors*M_btrmodel.ω).^2)
        mse_iters[em+1] = mse_new
        # Add coefficient updates
        ω_iters[:,em+1]= M_btrmodel.ω
        # Sync E and Mstep models
        display(join(["EM iteration ", em, " complete, MSE: ", mse_new]))
        E_btrmodel.ω = M_btrmodel.ω
        E_btrmodel.σ2 = M_btrmodel.σ2
        E_btrmodel.mse = mse_new
        M_btrmodel.mse = mse_new

        # Plot update on ω if specified
        if opts.plot_ω
            l = @layout [a{0.5h} b{0.5h}]
            plt = plot(layout = (2,1))
            plot!(0:em,transpose(ω_iters[:,1:(em+1)]),legend=false, subplot = 1,
            xticks = 0:em, xlab = "EM iteration", ylab = "Omega updates")
            plot!(0:em,mse_iters[1:(em+1)],legend=false, subplot = 2,
            xticks = 0:em, xlab = "EM iteration", ylab = "MSE updates")
            display(plt)
        end

        # If mse_conv > 0, then use this as convergence criterion
        if ((opts.mse_conv > 0) & (em > opts.mse_conv))
            mse_diff = (mean(mse_iters[em:(em-mse_conv+1)]) - mse_new)
            if mse_diff < 0
                btrmodel.converged = true
                break
            end
        else
            ω_diff = abs.(ω_iters[:,em+1] .- ω_iters[:,em])
            ω_diff[ω_diff.<opts.ω_tol] = zeros(sum(ω_diff.<opts.ω_tol))
            if opts.rel_tol
                ω_diff ./=abs.(M_btrmodel.ω)
            end
            if maximum(ω_diff) < opts.ω_tol
                btrmodel.converged = true
                break
            end
        end
    end # End of EM loop

    if opts.CVEM == :obs
        btrmodel = btrmodel_combineobs(btrmodel, E_btrmodel, M_btrmodel,E_obs, M_obs)
    end

    display("Computing in-sample perplexity")
    btrmodel.pplxy = compute_perplexity(btrmodel.crps, btrmodel.β, opts.α)

    return btrmodel
end


"""
BTC_EMGibbs plus interaction effects between x variables and topics in the regression
    order is (all topics)*x_1 + (all_topics)*x_2 etc...
"""
function BTCemGibbs(btcmodel::BTCModel)

    opts = btcmodel.options
    btcmodel.converged = false # Set convergence to false at the beginning

    ## Check that everything matches up
    @assert opts.ntopics == btcmodel.crps.ntopics "ntopics in corpus and options must match"
    @assert all(in.(opts.interactions, [opts.xregs])) "Can't have interactions that aren't specified in xreg"
    @assert btcmodel.crps.V == length(btcmodel.crps.vocab) "Vocab length in BTRModel and V in BTRCorpus do not match"
    @assert length(btcmodel.ω) == (opts.ntopics + length(opts.interactions)*opts.ntopics +
        (length(opts.xregs) - length(opts.interactions))) "Dimensions of ω do not match with xregs and interactions"

    # Make necessary changes if using Cross-Validation EM
    if opts.CVEM == :obs
        E_btcmodel, M_btcmodel, E_obs, M_obs = btcmodel_splitobs(btcmodel, opts.CVEM_split)
    else
        E_btcmodel = btcmodel
        M_btcmodel = btcmodel
        E_obs = 1:btcmodel.crps.N
        M_obs = 1:btcmodel.crps.N
    end
    # Initialise crossent with p = 0.5
    p_pred = repeat([0.5], length(M_btcmodel.y))
    btcmodel.crossent_iters[1] = cross_entropy(M_btcmodel.y, p_pred)

    # Begin EM loop
    for em in 1:opts.EM_iters
        ω_iters = btcmodel.ω_iters::Array{Float64,2}
        crossent_iters = btcmodel.crossent_iters::Array{Float64,1}
        crossent_conv = opts.crossent_conv::Int64
        display(join(["E step ", em]))
        E_btcmodel = BTCEstep(E_btcmodel)
        #heatmap(E_btcmodel.β)
        M_btcmodel.β = E_btcmodel.β

        display(join(["M step ", em]))
        if opts.CVEM == :obs
            M_btcpredict = BTCpredict(M_btcmodel.crps, E_btcmodel)
            M_btcmodel.Z_bar = Matrix(transpose(M_btcpredict.Z_bar))
            M_btcmodel.regressors = M_btcpredict.regressors
            M_btcmodel.crps = M_btcpredict.crps
        end
        M_btcmodel = BTCMstep(M_btcmodel)
        # Add new crossent to crossent_iters
        e_ylin = exp.(M_btcmodel.regressors*M_btcmodel.ω)::Array{Float64,1}
        p_pred = e_ylin./(1 .+ e_ylin)::Array{Float64,1}
        crossent_new = cross_entropy(M_btcmodel.y, p_pred)
        crossent_iters[em+1] = crossent_new
        # Add coefficient updates
        ω_iters[:,em+1]= M_btcmodel.ω

        # Synch up E and M-step models
        display(join(["EM iteration ", em, " complete, cross-entropy: ", crossent_new]))
        E_btcmodel.ω = M_btcmodel.ω
        M_btcmodel.crossent = crossent_new
        E_btcmodel.crossent = crossent_new

        # Plot update on ω if specified
        if opts.plot_ω
            l = @layout [a{0.5h} b{0.5h}]
            plt = plot(layout = (2,1))
            plot!(0:em,transpose(ω_iters[:,1:(em+1)]),legend=false, subplot = 1,
            xticks = 0:em, xlab = "EM iteration", ylab = "Omega updates")
            plot!(0:em,crossent_iters[1:(em+1)],legend=false, subplot = 2,
            xticks = 0:em, xlab = "EM iteration", ylab = "Cross-entropy updates")
            display(plt)
        end

        # If mse_conv > 0, then use this as convergence criterion
        if ((crossent_conv > 0) & (em > crossent_conv))
            crossent_diff = (mean(crossent_iters[em:(em-crossent_conv+1)]) - crossent_new)
            if crossent_diff <= 0
                btcmodel.converged = true
                break
            end
        else
            ω_diff = abs.(ω_iters[:,em+1] .- ω_iters[:,em])
            ω_diff[ω_diff.<opts.ω_tol] = zeros(sum(ω_diff.<opts.ω_tol))
            if opts.rel_tol
                ω_diff ./=abs.(M_btcmodel.ω)
            end
            if maximum(ω_diff) < opts.ω_tol
                btcmodel.converged = true
                break
            end
        end
    end # End of EM-loop

    if opts.CVEM == :obs
        btcmodel = btcmodel_combineobs(btcmodel, E_btcmodel, M_btcmodel,E_obs, M_obs)
    end

    display("Computing in-sample perplexity")
    btcmodel.pplxy = compute_perplexity(btcmodel.crps, btcmodel.β, opts.α)

    return btcmodel
end


"""
Split BTRModel for CVEM approach
"""
function btrmodel_splitobs(btrmodel::BTRModel, split_ratio::Float64)
    # Randomly split into E and M observations
    idx = shuffle(1:btrmodel.crps.N)
    # Separate indices into test and training
    E_obs = Array{Int64,1}(view(idx, 1:floor(Int, split_ratio*btrmodel.crps.N)))::Array{Int64,1}
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
    M_crps.N = length(M_crps.docs)
    # Other elements
    M_btrmodel.y = M_btrmodel.y[M_obs]
    M_btrmodel.regressors = M_btrmodel.regressors[M_obs,:]
    M_btrmodel.Z_bar = M_btrmodel.Z_bar[:,M_obs]

    return E_btrmodel,M_btrmodel,E_obs,M_obs
end


"""
Split BTCModel for CVEM approach
"""
function btcmodel_splitobs(btcmodel::BTCModel, split_ratio::Float64)
    # Randomly split into E and M observations
    idx = shuffle(1:btcmodel.crps.N)
    # Separate indices into test and training
    E_obs = Array{Int64,1}(view(idx, 1:floor(Int, split_ratio*btcmodel.crps.N)))::Array{Int64,1}
    M_obs = setdiff(idx, E_obs)::Array{Int64,1}

    ## Extract btrmodel for E-step
    E_btcmodel = deepcopy(btcmodel)
    # Update corpus
    E_crps = E_btcmodel.crps
    E_crps.docs = E_crps.docs[E_obs]
    E_crps.topics = gettopics(E_crps.docs)
    E_crps.docidx_labels = E_crps.docidx_labels[E_obs]
    E_crps.N = length(E_crps.docs)
    # Other elements
    E_btcmodel.y = E_btcmodel.y[E_obs]
    E_btcmodel.regressors = E_btcmodel.regressors[E_obs,:]
    E_btcmodel.Z_bar = E_btcmodel.Z_bar[:,E_obs]

    ## Extract btrmodel for M-step
    M_btcmodel = deepcopy(btcmodel)
    # Update corpus
    M_crps = M_btcmodel.crps
    M_crps.docs = M_crps.docs[M_obs]
    M_crps.topics = gettopics(M_crps.docs)
    M_crps.docidx_labels = M_crps.docidx_labels[M_obs]
    M_crps.N = length(M_crps.docs)
    # Other elements
    M_btcmodel.y = M_btcmodel.y[M_obs]
    M_btcmodel.regressors = M_btcmodel.regressors[M_obs,:]
    M_btcmodel.Z_bar = M_btcmodel.Z_bar[:,M_obs]

    return E_btcmodel,M_btcmodel,E_obs,M_obs
end

"""
Combines E and M step separate models back into combined Bayesian Topic Regression Model
"""
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
    btrmodel.mse = M_btrmodel.mse
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


"""
Combines E and M step separate models back into combined Bayesian Topic Regression Model
"""
function btcmodel_combineobs(btcmodel::BTCModel, E_btcmodel::BTCModel, M_btcmodel::BTCModel,
        E_obs::Array{Int64,1}, M_obs::Array{Int64,1})
    # Randomly split into E and M observations
    N = E_btcmodel.crps.N + M_btcmodel.crps.N
    @assert N == maximum(vcat(E_obs, M_obs)) "E and M step subsample sizes don't match total data"
    @assert N == btcmodel.crps.N "E and M step subsample sizes don't match total data"
    @assert E_btcmodel.β == M_btcmodel.β "E and M step β don't match"
    @assert E_btcmodel.ω == M_btcmodel.ω "E and M step β don't match"

    # Update regression parameters
    btcmodel.ω = M_btcmodel.ω
    btcmodel.crossent = M_btcmodel.crossent
    btcmodel.Σ = M_btcmodel.Σ

    # Update corpus
    btcmodel.crps.docs[E_obs] = E_btcmodel.crps.docs
    btcmodel.crps.docs[M_obs] = M_btcmodel.crps.docs
    btcmodel.crps.topics = gettopics(btcmodel.crps.docs)
    btcmodel.β = E_btcmodel.β

    # Other elements
    btcmodel.Z_bar[:,E_obs] = E_btcmodel.Z_bar
    btcmodel.Z_bar[:,M_obs] = M_btcmodel.Z_bar
    btcmodel.regressors[E_obs,:] = E_btcmodel.regressors
    btcmodel.regressors[M_obs,:] = M_btcmodel.regressors

    return btcmodel
end
