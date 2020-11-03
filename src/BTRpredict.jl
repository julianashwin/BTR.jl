"""
Out-of-sample prediction for BTR
    Takes an (unseen) BTRCorpus and a BTRModel as arguments
    Outputs a BTRPrediction object
"""
function BTRpredict(btrcrps::DocStructs.BTRCorpus, btrmodel::BTRModel)::BTRPrediction

    # Extract the estimation options
    opts::BTROptions = btrmodel.options
    ntopics::Int64 = opts.ntopics
    E_iters::Int64 = opts.E_iters
    burnin::Int64 = opts.burnin
    interactions::Array{Int64,1} = opts.interactions
    nointeractions::Array{Int64,1} = setdiff(opts.xregs, opts.interactions)

    # Extract the estimated parameters and priors
    α::Float64 = opts.α
    β::Array{Float64,2} = btrmodel.β
    ω::Array{Float64,1} = btrmodel.ω

    # Check that everything matches up
    @assert (size(β,1) == ntopics) "β needs to match with ntopics"
    @assert (size(β,2) == btrcrps.V) "β needs to match with ntopics"
    @assert opts.ntopics == btrcrps.ntopics "ntopics in corpus and options must match"
    @assert all(in.(opts.interactions, [opts.xregs])) "Can't have interactions that aren't specified in xreg"
    @assert btrmodel.crps.V == length(btrmodel.vocab) "Vocab length in BTRModel and V in BTRCorpus do not match"
    @assert length(ω) == (opts.ntopics + length(opts.interactions)*opts.ntopics +
        (length(opts.xregs) - length(opts.interactions))) "Dimensions of ω do not match with xregs and interactions"


    # Extract the data
    docs::Array{DocStructs.BTRParagraphDocument,1} = btrcrps.docs

    # Some placeholders
    zprobs::Array{Float64,1} = Vector{Float64}(undef, ntopics)
    topic_counts::Array{Array{Int64,1},1} = Array{Array{Int64,1},1}(undef,length(docs))
    Z_bar::Array{Float64,2} = zeros(ntopics,length(docs))
    Z_bar_avg::Array{Float64,2} = zeros(ntopics,length(docs))

    display("Sampling topic assignments")
    prog = Progress(E_iters+burnin, 1)
    for it in 1:(E_iters+burnin)
        ## Go through all the docs and assign topic to each word
        for (dd,doc) in enumerate(docs)
            paras = doc.paragraphs
            for para in paras
                # Remove paragraph counts from document counts
                doc.topicidcount -= para.topicidcount
                for (nn, word) in enumerate(para.text)
                    β_word::Array{Float64,1} = β[:,word]
                    topicid_current::Int64 = para.topic[nn] # Current id of this word
                    para.topicidcount[topicid_current] -= 1 # Remove this word from doc topic counts

                    for kk in 1:ntopics
                        term1::Float64  = (para.topicidcount[kk] + α) # prob of topic based on rest of doc
                        zprobs[kk] = β_word[kk]* (doc.topicidcount[kk] + α)
                    end
                    normalize_probs::Float64 = sum(zprobs)

                    # select new topic
                    selectz::Float64 = rand() # Draw from unit uniform (faster than Multinomial function from Distributions.jl)
                    sum_of_prob::Float64 = 0.0
                    new_topicid::Int64 = 0
                    for (selected_topicid, prob) in enumerate(zprobs)
                        sum_of_prob += prob / normalize_probs # Add normalised probability to sum
                        if selectz < sum_of_prob
                            new_topicid = selected_topicid # Select topic if sum greater than draw
                            break
                        end
                    end
                    #display(join(string.([dd,":",nn])))
                    para.topic[nn] = new_topicid # Populate document topic
                    para.topicidcount[new_topicid] = get(para.topicidcount, new_topicid, 0) + 1 # Add to topic counts

                end # End of word-level loop
                # Add updated paragraph counts back to document counts
                doc.topicidcount += para.topicidcount
            end # End of paragraph-level loop

        end # End of document-level loop
        # Extract Z_bar
        topic_counts = getfield.(docs, :topicidcount)
        Z_bar = Float64.(hcat(topic_counts...))
        Z_bar ./= sum(Z_bar, dims=1)
        if opts.emptydocs == :prior
            replace_nan!(Z_bar, 1/ntopics)
        elseif opts.emptydocs == :zero
            replace_nan!(Z_bar, 0.)
        end

        if it > burnin
            Z_bar_avg += (Z_bar.*(1/E_iters))::Array{Float64,2}
        end

        next!(prog)
    end # End of Gibbs sampled iterations

    # Create regressors for prediction
    y::Array{Float64,1} = vcat(getfield.(btrcrps.docs, :y)...)
    x::Array{Float64,2} = vcat(getfield.(btrcrps.docs, :x)...)
    Z_bar = Matrix(transpose(Z_bar_avg))
    # Placeholders to store the document-topic and topic-word distributions
    inter_effects::Array{Float64,2} = zeros(btrcrps.N,(ntopics*length(interactions)))
    for jj in 1:length(interactions)
        col_range::UnitRange{Int64} = ((jj*ntopics-ntopics+1):(jj*ntopics))
        inter_effects[:,col_range] = Z_bar.*vec(x[:,interactions[jj]])
    end
    regressors::Array{Float64,2} = hcat(Z_bar, inter_effects, x[:,nointeractions])
    @assert size(regressors) == (btrcrps.N, length(ω)) "Regressors and ω don't match"

    # Renormalise θ and β to eliminate any numerical errors
    y_pred::Array{Float64,1} = regressors*ω

    predictions = BTRPrediction(options = opts, crps = btrcrps, Z_bar = Z_bar, y_pred = y_pred,
        regressors = regressors, y = y)
    return predictions

end
