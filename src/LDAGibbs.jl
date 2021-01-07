"""
Function that estimates a standard LDA model on a BTRCorpus with (collapsed) Gibbs sampling
"""
function LDAGibbs(ldamodel::Union{BTRModel,BTCModel})::Union{BTRModel,BTCModel}
    ## Extract the important parts of btrmodel
    docs::Array{Union{DocStructs.BTRParagraphDocument,DocStructs.BTCParagraphDocument},1} =
        ldamodel.crps.docs
    topics::Array{DocStructs.Topic,1} = ldamodel.crps.topics
    opts::Union{BTROptions,BTCOptions} = ldamodel.options
    xregs::Array{Int64,1} = opts.xregs
    interactions::Array{Int64,1} = opts.interactions
    nointeractions::Array{Int64,1} = setdiff(xregs, interactions)
    ## Some useful identifiers and parameters
    ntopics::Int64 = opts.ntopics
    V = ldamodel.crps.V::Int64
    α::Float64 = opts.α
    η::Float64 = opts.η

    # Check that everything matches up
    @assert opts.ntopics == ldamodel.crps.ntopics "ntopics in corpus and options must match"
    @assert ldamodel.crps.V == length(ldamodel.vocab) "Vocab length in BTRModel and V in BTRCorpus do not match"

    # Placeholders to store assignments for each iteration
    β::Array{Float64,2} = zeros(ntopics,V)
    Z_bar::Array{Float64,2} = zeros(ntopics,ldamodel.crps.N)
    zprobs::Array{Float64,1} = zeros(ntopics)

    ## Set Z_bar and β to zero to be filled with the average across iterations
    ldamodel.Z_bar::Array{Float64,2} = zeros(size(ldamodel.Z_bar))
    ldamodel.β::Array{Float64,2} = zeros(size(ldamodel.β))

    ## Gibbs sampling
    n_samples = floor(Int, opts.fullGibbs_iters/opts.fullGibbs_thinning)
    display("Starting Gibbs Sampling")
    prog = Progress(opts.fullGibbs_iters+opts.burnin, 1)
    for it in 1:(opts.fullGibbs_iters+opts.burnin)
        ## Go through all the docs and assign topic to each word
        for (dd,doc) in enumerate(docs)

            # Sum of words across the *document*
            paras::Array{DocStructs.TopicBasedDocument,1} = doc.paragraphs
            for para in paras
                ## What would the prediction for y be without this paragraph?
                doc.topicidcount -= para.topicidcount

                for (nn, word) in enumerate(para.text)
                    topicid_current = para.topic[nn] # Current id of this word
                    para.topicidcount[topicid_current] -= 1 # Remove this word from doc topic counts
                    topics[topicid_current].count -= 1 # Remove from overall topic count
                    topics[topicid_current].wordcount[word] -= 1 # Remove from word-topic count

                    # Residual for y given all other topic assignments
                    for kk in 1:ntopics
                        term1 = (para.topicidcount[kk] + α) # prob of topic based on rest of doc
                        term2 = (get(topics[kk].wordcount, word, 0)+ η) / (topics[kk].count + η * V) # prob of word based on topic
                        zprobs[kk] = term1 * term2

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
                    if new_topicid < 1
                        display(new_topicid)
                    end
                    para.topicidcount[new_topicid] = get(para.topicidcount, new_topicid, 0) + 1 # Add to topic counts
                    topics[new_topicid].count += 1 # Add to total topic count
                    topics[new_topicid].wordcount[word] = get(topics[new_topicid].wordcount, word, 0) + 1 # Add to topic-word count
                end ## End of word-level cycle
                doc.topicidcount += para.topicidcount

            end ## End of paragraph-level cycle

        end ## End of document-level cycle

        ## Update corpus-level Z_bar
        Z_bar = Float64.(hcat(getfield.(docs, :topicidcount)...))
        Z_bar ./= sum(Z_bar, dims=1)
        if opts.emptydocs == :prior
            replace_nan!(Z_bar, 1/ntopics)
        elseif opts.emptydocs == :zero
            replace_nan!(Z_bar, 0.)
        end

        ## Update β topic vectors
        for topic in 1:ntopics
            t = topics[topic]
            β[topic, :] .= (η) / (t.count + V*η)
            for (word, count) in t.wordcount
                if count > 0
                    β[topic, word] = (count + η) / (t.count + V*η)
                end
            end
        end


        if it > opts.burnin && it%opts.fullGibbs_thinning ==0
            ldamodel.β += (β.*(1/opts.E_iters))::Array{Float64,2}
            ldamodel.Z_bar += (Z_bar.*(1/opts.E_iters))::Array{Float64,2}
        end
        next!(prog)
    end ## End of E-step cycle


    # Create the regressors
    Z_bar = Matrix(transpose(ldamodel.Z_bar))
    x = vcat(getfield.(ldamodel.crps.docs, :x)...)
    inter_effects = zeros(length(docs), ntopics*length(interactions))::Array{Float64,2}
    for jj in 1:length(opts.interactions)
        col_range = ((jj*ntopics-ntopics+1):(jj*ntopics))
        inter_effects[:,col_range] = Z_bar.*vec(x[:,interactions[jj]])
    end
    ldamodel.regressors = hcat(Z_bar, inter_effects, x[:,nointeractions])

    # Renormalise θ and β to eliminate any numerical errors
    ldamodel.Z_bar ./= sum(ldamodel.Z_bar, dims=1)::Array{Float64,2}
    ldamodel.β ./= sum(ldamodel.β, dims=2)::Array{Float64,2}

    return ldamodel
end
