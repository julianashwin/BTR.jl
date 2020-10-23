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


    for em in 1:opts.EM_iters
        ω_iters = btrmodel.ω_iters::Array{Float64,2}
        display(join(["E step ", em]))
        btrmodel = BTREstep(btrmodel)
        heatmap(btrmodel.β)

        display(join(["M step ", em]))
        btrmodel = BTRMstep(btrmodel)
        display(join(["EM iteration ", em, " complete, MSE: ", btrmodel.σ2]))

        # Find coefficient updates
        ω_iters[:,em+1]= btrmodel.ω
        ω_diff = abs.(ω_iters[:,em+1] .- ω_iters[:,em])
        ω_diff[ω_diff.<opts.ω_tol] = zeros(sum(ω_diff.<opts.ω_tol))
        if opts.rel_tol
            ω_diff ./=abs.(btrmodel.ω)
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

    return btrmodel
end
