"""
Function that performs the E-step, of sampling topic assignments, on a BTRModel object
"""
function BTREstep(btrmodel::BTRModel)
    ## Extract the important parts of btrmodel
    docs::Array{DocStructs.BTRParagraphDocument,1} = btrmodel.crps.docs
    topics::Array{DocStructs.Topic,1} = btrmodel.crps.topics
    opts::BTROptions = btrmodel.options
    ## Some useful identifiers and parameters
    ntopics::Int64 = opts.ntopics
    V::Int64 = btrmodel.crps.V
    interactions::Array{Int64,1} = opts.interactions
    nointeractions::Array{Int64,1} = setdiff(opts.xregs, opts.interactions)
    α::Float64 = opts.α
    η::Float64 = opts.η
    σ2::Float64 = btrmodel.σ2
    ## Split up ω into the relevant parts
    ω_z::Array{Float64,1} = btrmodel.ω[1:ntopics]
    ω_zx::Array{Float64,1} = btrmodel.ω[(ntopics+1):(ntopics+length(interactions)*ntopics)]
    ω_x::Array{Float64,1} = btrmodel.ω[(ntopics+length(interactions)*ntopics+1):length(btrmodel.ω)]
    ω_docspec::Array{Float64,1} = zeros(size(btrmodel.ω))
    ## Objects used as placeholders
    zprobs::Array{Float64,1} = zeros(ntopics)
    y_dd::Float64 = 0.
    ypred_minusd::Float64 = 0.
    ypred_minusp::Float64 = 0.
    β::Array{Float64,2} = zeros(ntopics,V)
    Z_bar::Array{Float64,2} = zeros(ntopics,btrmodel.crps.N)
    ## Set Z_bar and β to zero to be filled with the average across iterations
    btrmodel.Z_bar::Array{Float64,2} = zeros(size(btrmodel.Z_bar))
    btrmodel.β::Array{Float64,2} = zeros(size(btrmodel.β))

    ## Loop through the Gibbs sampler iterations
    prog = Progress(opts.E_iters+opts.burnin, 1)
    for it in 1:(opts.E_iters+opts.burnin)
        ## Go through all the docs and assign topic to each word
        for (dd,doc) in enumerate(docs)

            ## First up gather necessary metadata
            y_dd = doc.y

            # Predicted y without the document (i.e. just x)
            ypred_minusd = sum(ω_x.*doc.x[:,nointeractions])

            ## The document specific ω, based on the interaction effects
            ω_docspec = zeros(ntopics)::Array{Float64,1}
            ω_docspec += ω_z
            # add each interaction effect to the document specific ω
            for jj in 1:length(interactions)
                col_range::UnitRange{Int64} = (jj*ntopics-ntopics+1):(jj*ntopics)
                ω_docspec.+= ω_zx[col_range].*doc.x[:,interactions[jj]]
            end

            # Sum of words across the *document*
            N_d::Int64 = sum(doc.topicidcount)
            paras = doc.paragraphs::Array{DocStructs.TopicBasedDocument,1}
            for para in paras
                ## What would the prediction for y be without this paragraph?
                doc.topicidcount -= para.topicidcount
                ypred_minusp = ypred_minusd + sum((ω_docspec./N_d).*doc.topicidcount)

                for (nn, word) in enumerate(para.text)
                    topicid_current = para.topic[nn] # Current id of this word
                    para.topicidcount[topicid_current] -= 1 # Remove this word from doc topic counts
                    topics[topicid_current].count -= 1 # Remove from overall topic count
                    topics[topicid_current].wordcount[word] -= 1 # Remove from word-topic count

                    # Residual for y given all other topic assignments
                    res = (y_dd - ypred_minusp - sum(ω_docspec.*para.topicidcount)./N_d)
                    for kk in 1:ntopics
                        term1 = (para.topicidcount[kk] + α) # prob of topic based on rest of doc
                        term2 = (get(topics[kk].wordcount, word, 0)+ η) / (topics[kk].count + η * V) # prob of word based on topic
                        term3 = exp((1/(2*σ2))*((2*ω_docspec[kk]/N_d)*res -  (ω_docspec[kk]/N_d)^2)) # predictive distribution for y
                        zprobs[kk] = term1 * term2 * term3

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

        if it > opts.burnin
            btrmodel.β += (β.*(1/opts.E_iters))::Array{Float64,2}
            btrmodel.Z_bar += (Z_bar.*(1/opts.E_iters))::Array{Float64,2}
        end
        next!(prog)
    end ## End of E-step cycle

    return btrmodel
end
