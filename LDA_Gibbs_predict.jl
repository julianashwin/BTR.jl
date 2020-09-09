

"""
Out-of-sample prediction for BTR
"""
function LDA_Gibbs_predict(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int,
    β::Array{Float64,2};
    α::Float64 =1.,
    iteration::Int64 = 1000, burnin::Int64 = 100)

    @assert (size(β,1) == ntopics) "β needs to match with ntopics"

    # Initialise variables
    D, V = size(dtm_in)
    docs = Array{Lda.TopicBasedDocument}(undef, D)
    topics = Array{Lda.Topic}(undef, ntopics)
    for i in 1:ntopics
        topics[i] = Lda.Topic()
    end
    total_samples = iteration + burnin

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

    # Gibbs sampling topic assignments
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
            θ_avg += (θ.*(1/iteration))::Array{Float64,2}
            Z_bar_avg += (Z_bar.*(1/iteration))::Array{Float64,2}
        end

        next!(prog)

    end # End of Gibbs sampled iterations

    # Renormalise θ and β to eliminate any numerical errors
    Z_bar_avg ./= sum(Z_bar_avg, dims=1)::Array{Float64,2}
    θ_avg ./= sum(θ_avg, dims=1)::Array{Float64,2}

    return Z_bar_avg

end
