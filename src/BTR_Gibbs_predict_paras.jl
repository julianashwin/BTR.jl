

"""
Out-of-sample prediction for BTR
"""
function BTR_Gibbs_predict_paras(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int,
    β::Array{Float64,2}, ω::Array{Float64,1};
    x::Array{Float64,2} = zeros(1,1),
    Σ::Array{Float64,2} = zeros(2,2),
    doc_idx::Array{Int64,1} = [0],
    σ2::Float64 = 0., y::Array{Float64,1} = zeros(1),
    α::Float64 =1., 
    E_iteration::Int64 = 100, burnin::Int64 = 10,
    interactions::Array{Int64,1}=Array{Int64,1}([]))

    @assert (size(β,1) == ntopics) "β needs to match with ntopics"

    # Extract relevant information about paragraphs etc
    if length(doc_idx)==size(dtm_in,1)
        display("Estimating with paragraphs/sentences")
    else
        display("Estimating without paragraphs/sentences")
        doc_idx = Array(1:size(dtm_in,1))
    end

    # Initialise variables
    D, V = size(dtm_in)
    docs = Array{Lda.TopicBasedDocument}(undef, D)
    topics = Array{Lda.Topic}(undef, ntopics)
    for i in 1:ntopics
        topics[i] = Lda.Topic()
    end
    mse_oos::Float64 = 0.0
    total_samples::Int64 = E_iteration + burnin

    # Deal with no x case and pre-specified σ2
    if size(x,1) == D
        J = size(x,2)
        @assert length(interactions) <= J "Can't have more interactions than x variables"
        J_inter = length(interactions)*ntopics
        J_nointer = J - length(interactions)
        no_interactions = setdiff(1:J, interactions)
    else
        J = 0
        J_inter = 0
        J_nointer = 0
        no_interactions = Array{Int64,1}([])
    end

    # Extract the documents and initialise the topic assignments
    for dd in 1:D
        topic_base_document = Lda.TopicBasedDocument(ntopics)
        for wordid in 1:V
            for _ in 1:dtm_in[dd,wordid]
                topicid = rand(1:ntopics) # initial topic assignment
                update_target_topic = topics[topicid] # select relevant topic
                update_target_topic.count += 1 # add to total words in that topc
                update_target_topic.wordcount[wordid] = get(update_target_topic.wordcount, wordid, 0) + 1 # add to count for that word
                topics[topicid] = update_target_topic # update that topic
                push!(topic_base_document.topic, topicid) # add topic to document
                push!(topic_base_document.text, wordid) # add word to document
                topic_base_document.topicidcount[topicid] =  get(topic_base_document.topicidcount, topicid, 0) + 1 # add topic to topic count for document
            end
        end
        docs[dd] = topic_base_document # Populate the document array
    end

    # Vector for topic probabilities
    probs = Vector{Float64}(undef, ntopics)
    # Vector of topic counts
    topic_counts = getfield.(docs, :topicidcount)

    # Placeholders to store the document-topic and topic-word distributions
    θ = zeros(ntopics,D)
    θ_avg = zeros(ntopics,D)
    # Optional placeholder for the average Z in each document across iterations
    Z_bar = zeros(ntopics,D)
    Z_bar_avg = zeros(ntopics,D)
    inter_effects = zeros(D,(J_inter))::Array{Float64,2}::Array{Float64,2}
    regressors = zeros(D, length(ω))::Array{Float64,2}

    # Gibbs sampling E-step
    display(join(["Gibbs Sampling for topic assignments"]))
    #@showprogress 1
    prog = Progress(total_samples, 1)
    for it in 1:total_samples
        # Run through all documents, assigning z
        for (dd,doc) in enumerate(docs)
            # Run through each word in doc
            for (nn, word) in enumerate(doc.text)
                topicid_current = doc.topic[nn] # Current id of this word
                doc.topicidcount[topicid_current] -= 1 # Remove this word from doc topic count counts
                topics[topicid_current].count -= 1 # Remove from overall topic count
                topics[topicid_current].wordcount[word] -= 1 # Remove from word-topic count
                N_d = length(doc.text) # Document length (minus this word)

                for kk in 1:ntopics
                    term1 = (doc.topicidcount[kk] + α) # prob of topic based on rest of doc
                    probs[kk] = β[kk,word]* (doc.topicidcount[kk] + α)
                    if probs[kk] == 0.0
                        probs[kk] = (doc.topicidcount[kk] + α)
                    end
                end
                normalize_probs = sum(probs)

                # select new topic
                select = rand() # Draw from unit uniform (faster than Multinomial function from Distributions.jl)
                sum_of_prob = 0.0
                new_topicid = 0
                for (selected_topicid, prob) in enumerate(probs)
                    sum_of_prob += prob / normalize_probs # Add normalised probability to sum
                    if select < sum_of_prob
                        new_topicid = selected_topicid # Select topic if sum greater than draw
                        break
                    end
                end
                #display(join(string.([dd,":",nn])))
                doc.topic[nn] = new_topicid # Populate document topic
                if new_topicid < 1
                    display(new_topicid)
                end
                doc.topicidcount[new_topicid] = get(doc.topicidcount, new_topicid, 0) + 1 # Add to topic counts
                topics[new_topicid].count += 1 # Add to total topic count
                topics[new_topicid].wordcount[word] = get(topics[new_topicid].wordcount, word, 0) + 1 # Add to topic-word count
            end
        end
        # Calculate the β, θ and Z_avg for this iteration
        topic_counts = getfield.(docs, :topicidcount)
        θ = Float64.(hcat(topic_counts...)) .+ α
        θ ./= sum(θ, dims=1)
        Z_bar = Float64.(hcat(topic_counts...))
        Z_bar ./= sum(Z_bar, dims=1)
        replace_nan!(Z_bar, 1/ntopics)

        ### Add contribution to average β, θ and Z_avg
        if it > burnin
            θ_avg += (θ.*(1/E_iteration))::Array{Float64,2}
            Z_bar_avg += (Z_bar.*(1/E_iteration))::Array{Float64,2}
        end

        next!(prog)

    end # End of Gibbs sampled iterations

    # Renormalise θ and β to eliminate any numerical errors
    Z_bar_avg ./= sum(Z_bar_avg, dims=1)::Array{Float64,2}
    θ_avg ./= sum(θ_avg, dims=1)::Array{Float64,2}
    @assert any(Z_bar_avg.!= 1/ntopics) "Need some observations with documents"

    # Predict y (Z first, then x)
    if J > 0
        if length(interactions) == 0
            regressors = hcat(transpose(Z_bar_avg), x)::Array{Float64,2}
        else
            for jj in 1:length(interactions)
                col_range= (jj*ntopics-ntopics+1):(jj*ntopics)
                inter_effects[:,col_range] = transpose(Z_bar_avg).*vec(x[:,interactions[jj]])
            end
            regressors = hcat(transpose(Z_bar_avg), inter_effects,x[:,no_interactions])::Array{Float64,2}
        end
    else
        regressors = hcat(transpose(Z_bar_avg))::Array{Float64,2}
    end

    y_pred = regressors*ω
    if size(y,1) == D
        mse_oos = mean((y .- y_pred).^2)
    end

    return (y_pred = y_pred, θ = θ_avg, Z_bar = Z_bar_avg,
        mse_oos = mse_oos, docs = docs, topics = topics)

end
