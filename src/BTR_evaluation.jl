"""
Compute perplexity from BT(R/C)Corpus, β and α
"""
function compute_perplexity(crps::Union{DocStructs.BTRCorpus,DocStructs.BTCCorpus},
    β::Array{Float64,2}, α::Float64)::Float64

    ntopics = crps.ntopics
    @assert ntopics == size(β,1) "ntopics in crps and β need to match"


    pplxy::Float64 = 0.

    # Extract necessary objects
    docs = crps.docs
    NP::Int64 = sum(length.(getfield.(docs, :paragraphs)))
    N_words::Int64 = sum(length.(getfield.(vcat(getfield.(docs, :paragraphs)...),:text)))

    # Loop through to compute log-likelihood of each paragraph
    prog = Progress(length(docs), 1)
    llk::Array{Float64,1} = zeros(NP)
    pp::Int64 = 1
    for (dd,doc) in enumerate(docs)

        paras = doc.paragraphs
        for para in paras
            θ_p::Array{Float64,1} = (para.topicidcount .+ α)
            θ_p ./= sum(θ_p, dims=1)

            for (nn, word) in enumerate(para.text)
                β_word::Array{Float64,1} = β[:,word]
                llk[pp] +=  log(sum(θ_p.*β_word)) # log(p(w|θ,β))
            end # end word loop
            pp +=1
        end # end para loop
        next!(prog)
    end # end doc loop

    # Compute combined perplexity
    pplxy = exp(sum(llk)/N_words)

    return pplxy
end




"""
Function for multiple runs at out of sample prediction with Bayesian Topic Regression
"""
function BTR_multipleruns(train_data::DocStructs.BTRRawData, test_data::DocStructs.BTRRawData,
    opts::BTROptions, nruns::Int64, subdirectory::String, save_jld::Bool = false)

    performance_across_runs = DataFrame(run = 1:nruns, mse = repeat([0.], nruns),
        pplxy = repeat([0.], nruns))


    for nn in 1:nruns
        display(join(["Running for the ", nn, "th time"]))
        # Create the training and test corpora
        btrcrps_tr = create_btrcrps(train_data, opts.ntopics)
        btrcrps_ts = create_btrcrps(test_data, opts.ntopics)

        # Train on the training set
        btrmodel = BTRModel(crps = btrcrps_tr, options = opts)
        btrmodel = BTRemGibbs(btrmodel)

        # Out of sample prediction
        btr_predicts = BTRpredict(btrcrps_ts, btrmodel)

        ### Extract the info to be saved
        # Prediction data
        data_df = DataFrame(hcat(btr_predicts.regressors, test_data.y,
            btr_predicts.y_pred))
        # Define interaction labels
        nointeractions = setdiff(opts.xregs, opts.interactions)
        interaction_labels = []
        for jj in opts.interactions
            for kk in 1:opts.ntopics
                push!(interaction_labels, join([string(kk), string(jj)]))
            end
        end
        # Name columns
        rename!(data_df, vcat([Symbol("Zbar$i") for i in 1:opts.ntopics],
            [Symbol("ZbarX$i") for i in interaction_labels],
            [Symbol("X$i") for i in nointeractions], [:y,:y_pred]))
        # Estimated topics
        topics_df = DataFrame(hcat(btcmodel.crps.vocab,transpose(btcmodel.β)))
        rename!(topics_df, vcat(:term,[Symbol("T$k") for k in 1:opts.ntopics]))
        # Estimation options
        opts_df = DataFrame(ntopics = opts.ntopics, xregs = join(opts.xregs, ", "),
            interactions = join(opts.interactions, ", "), E_iters = opts.E_iters,
            M_iters = opts.M_iters, EM_iters = opts.EM_iters, burnin = opts.burnin,
            ω_tol = opts.ω_tol, rel_tol = opts.rel_tol, CVEM = opts.CVEM,
            CVEM_split = opts.CVEM_split, alpha = opts.α, eta = opts.η,
            mu_omega = opts.μ_ω, sigma_omega = opts.σ_ω, a_0 = opts.a_0, b_0 = opts.b_0)

        coef_df = DataFrame(coef = ["omega$i" for i in 1:length(btrmodel.ω)],
            est = btrmodel.ω)
        # Estimation output/performance
        mse_pred = mean((test_data.y .- btr_predicts.y_pred).^2)
        pplxy_pred = btr_predicts.pplxy
        performance_df = DataFrame(mse = mse_pred, pplxy = pplxy_pred)

        # Add performance stats into the summary table
        performance_across_runs[nn,:mse] = mse_pred
        performance_across_runs[nn,:pplxy] = pplxy_pred

        ### Save the files in specified subdirectory
        export_dir = join([subdirectory, nn])
        if !isdir(export_dir)
            mkpath(export_dir)
        end
        CSV.write(join([export_dir "/data_df.csv"]), data_df)
        CSV.write(join([export_dir "/topics_df.csv"]), topics_df)
        CSV.write(join([export_dir "/coef_df.csv"]), coef_df)
        CSV.write(join([export_dir "/opts_df.csv"]), opts_df)
        CSV.write(join([export_dir "/performance_df.csv"]), performance_df)

        if save_jld
            save(join([export_dir "/model.jld2"]), "model", btcmodel)
            save(join([export_dir "/predictions.jld2"]), "predictions", btc_predicts)
        end

    end

        # Save the summary performance across all runs
        CSV.write(join([subdirectory, "performance_summary.csv"]), performance_across_runs)

end




"""
Function for multiple runs at out of sample prediction with Bayesian Topic Classification
"""
function BTC_multipleruns(train_data::DocStructs.BTCRawData, test_data::DocStructs.BTCRawData,
    opts::BTCOptions, nruns::Int64, subdirectory::String, save_jld::Bool = false)

    performance_across_runs = DataFrame(run = 1:nruns, mse = repeat([0.], nruns),
        correct = repeat([0.], nruns), pplxy = repeat([0.], nruns))


    for nn in 1:nruns
        display(join(["Running for the ", nn, "th time"]))
        # Create the training and test corpora
        btccrps_tr = create_btccrps(train_data, opts.ntopics)
        btccrps_ts = create_btccrps(test_data, opts.ntopics)

        # Train on the training set
        btcmodel = BTCModel(crps = btccrps_tr, options = opts)
        btcmodel = BTCemGibbs(btcmodel)

        # Out of sample prediction
        btc_predicts = BTCpredict(btccrps_ts, btcmodel)

        ### Extract the info to be saved
        # Prediction data
        data_df = DataFrame(hcat(btc_predicts.regressors, test_data.y,
            btc_predicts.y_pred, btc_predicts.p_pred))
        nointeractions = setdiff(opts.xregs, opts.interactions)
        interaction_labels = []
        for jj in opts.interactions
            for kk in 1:opts.ntopics
                push!(interaction_labels, join([string(kk), string(jj)]))
            end
        end
        # Name columns
        rename!(data_df, vcat([Symbol("Zbar$i") for i in 1:opts.ntopics],
            [Symbol("ZbarX$i") for i in interaction_labels],
            [Symbol("X$i") for i in nointeractions], [:y,:y_pred,:p_pred]))
        # Estimated topics
        topics_df = DataFrame(hcat(btcmodel.crps.vocab,transpose(btcmodel.β)))
        rename!(topics_df, vcat(:term,[Symbol("T$k") for k in 1:opts.ntopics]))
        # Estimation options
        opts_df = DataFrame(ntopics = opts.ntopics, xregs = join(opts.xregs, ", "),
            interactions = join(opts.interactions, ", "), E_iters = opts.E_iters,
            M_iters = opts.M_iters, EM_iters = opts.EM_iters, burnin = opts.burnin,
            ω_tol = opts.ω_tol, rel_tol = opts.rel_tol, CVEM = opts.CVEM,
            CVEM_split = opts.CVEM_split, alpha = opts.α, eta = opts.η)

        coef_df = DataFrame(coef = ["omega$i" for i in 1:length(btcmodel.ω)],
            est = btcmodel.ω)
        # Estimation output/performance
        correct_pred = mean(test_data.y .== btc_predicts.y_pred)
        mse_pred = mean((test_data.y .- btc_predicts.p_pred).^2)
        pplxy_pred = btc_predicts.pplxy
        performance_df = DataFrame(mse = mse_pred, correct_prop = correct_pred,
            pplxy = pplxy_pred)

        # Add performance stats into the summary table
        performance_across_runs[nn,:mse] = mse_pred
        performance_across_runs[nn,:correct] = correct_pred
        performance_across_runs[nn,:pplxy] = pplxy_pred


        ### Save the files in specified subdirectory
        export_dir = join([subdirectory, nn])
        if !isdir(export_dir)
            mkpath(export_dir)
        end
        CSV.write(join([export_dir "/data_df.csv"]), data_df)
        CSV.write(join([export_dir "/topics_df.csv"]), topics_df)
        CSV.write(join([export_dir "/coef_df.csv"]), coef_df)
        CSV.write(join([export_dir "/opts_df.csv"]), opts_df)
        CSV.write(join([export_dir "/performance_df.csv"]), performance_df)

        if save_jld
            save(join([export_dir "/model.jld2"]), "model", btcmodel)
            save(join([export_dir "/predictions.jld2"]), "predictions", btc_predicts)
        end

    end

    # Save the summary performance across all runs
    CSV.write(join([subdirectory, "performance_summary.csv"]), performance_across_runs)

end
