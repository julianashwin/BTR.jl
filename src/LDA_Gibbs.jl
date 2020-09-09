"""
    β, θ = lda(dtm::DocumentTermMatrix, ntopics::Int, iterations::Int, α::Float64, β::Float64)
Perform [Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation).
# Arguments
- `α` Dirichlet dist. hyperparameter for topic distribution per document. `α<1` yields a sparse topic mixture for each document. `α>1` yields a more uniform topic mixture for each document.
- `β` Dirichlet dist. hyperparameter for word distribution per topic. `η<1` yields a sparse word mixture for each topic. `β>1` yields a more uniform word mixture for each topic.
# Return values
- `β`: `ntopics × nwords` Sparse matrix of probabilities s.t. `sum(β, 1) == 1`
- `θ`: `ntopics × ndocs` Dense matrix of probabilities s.t. `sum(θ, 1) == 1`
"""
function LDA_Gibbs(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int64;
    iteration::Int64 = 1000, α::Float64 = 1., η::Float64 = 1., burnin = 100)

    # Initialise variables
    D, V = size(dtm_in)
    docs = Array{Lda.TopicBasedDocument}(undef, D)
    topics = Array{Lda.Topic}(undef, ntopics)
    for i in 1:ntopics
        topics[i] = Lda.Topic()
    end
    total_samples::Int64 = iteration + burnin

    # Placeholders to store assignments for each iteration
    θ = zeros(ntopics,D)
    θ_avg = zeros(ntopics,D)
    β = zeros(ntopics,V)
    β_avg = zeros(ntopics,V)
    Z_bar = zeros(ntopics,D)
    Z_bar_avg = zeros(ntopics,D)

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
    # Gibbs sampling
    display("Starting Gibbs Sampling")
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
                    term2 = (get(topics[kk].wordcount, word, 0)+ η) / (topics[kk].count + η * V) # prob of word based on topic
                    probs[kk] = term1 * term2
                end
                normalize_probs = sum(probs)

                # select new topic
                select = rand()
                sum_of_prob = 0.0
                new_topicid = 0
                for (selected_topicid, prob) in enumerate(probs)
                    sum_of_prob += prob / normalize_probs
                    if select < sum_of_prob
                        new_topicid = selected_topicid
                        break
                    end
                end
                doc.topic[nn] = new_topicid
                doc.topicidcount[new_topicid] = get(doc.topicidcount, new_topicid, 0) + 1
                topics[new_topicid].count += 1
                topics[new_topicid].wordcount[word] = get(topics[new_topicid].wordcount, word, 0) + 1
            end
        end
        # Calculate the β, θ and Z_bar for this iteration
        topic_counts = getfield.(docs, :topicidcount)
        θ = Float64.(hcat(topic_counts...)) .+ α
        θ ./= sum(θ, dims=1)
        Z_bar = Float64.(hcat(topic_counts...))
        Z_bar ./= sum(Z_bar, dims=1)
        replace_nan!(Z_bar, 1/ntopics)

        for topic in 1:ntopics
            t = topics[topic]
            β[topic, :] .= (η) / (t.count + V*η)
            for (word, count) in t.wordcount
                if count > 0
                    β[topic, word] = (count + η) / (t.count + V*η)
                end
            end
        end
        ### Add contribution to average β, θ and Z_avg
        if it > burnin
            β_avg += (β.*(1/iteration))::Array{Float64,2}
            θ_avg += (θ.*(1/iteration))::Array{Float64,2}
            Z_bar_avg += (Z_bar.*(1/iteration))::Array{Float64,2}
        end


        next!(prog)
    end

    # Renormalise θ and β to eliminate any numerical errors
    Z_bar_avg ./= sum(Z_bar_avg, dims=1)::Array{Float64,2}
    θ_avg ./= sum(θ_avg, dims=1)::Array{Float64,2}
    β_avg ./= sum(β_avg, dims=2)::Array{Float64,2}

    return (β = β_avg, θ = θ_avg, Z_bar = Z_bar_avg, docs = docs, topics = topics)
end
