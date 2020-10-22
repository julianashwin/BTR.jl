"""
BTR_EMGibbs plus interaction effects between x variables and topics in the regression
    order is (all topics)*x_1 + (all_topics)*x_2 etc...
"""
function BTRfullGibbs(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int,
    y::Array{Float64,1};  x::Array{Float64,2} = zeros(1,1),
    σ2::Float64 = 0.,
    α::Float64 =1., η::Float64 = 1., σ_ω::Float64 = 1., μ_ω::Float64 = 0.,
    a_0::Float64 = 0., b_0::Float64  = 0.,
    E_iteration::Int = 100, M_iteration::Int = 500, EM_iteration::Int = 10,
    topics_init::Array = Array{Main.Lda.Topic,1}(undef, 1),
    docs_init::Array = Array{Main.Lda.TopicBasedDocument,1}(undef, 1),
    ω_init::Array{Float64,1} = zeros(1), ω_tol::Float64 = 0.01,
    interactions::Bool=false, batch::Bool=false, EM_split::Float64 = 0.75)

    error("Not ready yet")

    # Define EM split
    if batch
        E_idx = 1:Int(round(EM_split*size(dtm_in,1)))
        M_idx = Int(round(EM_split*size(dtm_in,1)))+1:size(dtm_in,1)
    else
        E_idx = 1:size(dtm_in,1)
        M_idx = 1:size(dtm_in,1)
    end
    if x != zeros(1,1)
        x_Estep = x[E_idx,:]
        x_Mstep = x[M_idx,:]
    else
        x_Estep = zeros(1,1)
        x_Mstep = zeros(1,1)
    end
    y_Estep = y[E_idx]
    y_Mstep = y[M_idx]
    dtm_Estep = dtm_in[E_idx,:]
    dtm_Mstep = dtm_in[M_idx,:]


    # Initialise variables
    D, V = size(dtm_Estep)
    docs::Array{Lda.TopicBasedDocument,1} = Array{Lda.TopicBasedDocument,1}(undef, D)
    topics::Array{Lda.Topic,1} = Array{Lda.Topic,1}(undef, ntopics)
    for i in 1:ntopics
        topics[i] = Lda.Topic()
    end

    # Deal with no x case and pre-specified σ2
    if size(x_Estep,1) == D
        J = size(x_Estep,2)
    else
        J = 0
    end
    if σ2 == 0.
        est_σ2 = true
        σ2 = var(y_Estep)
    else
        est_σ2 = false
    end

    # Regression variables
    #ω = μ_ω*ones(ntopics+J+(ntopics*J))::Array{Float64,1}
    if interactions
        ω = μ_ω*ones(ntopics+(ntopics*J))::Array{Float64,1}
        ω_post = zeros(ntopics+(ntopics*J), M_iteration)::Array{Float64,2}
        ω_zx = ω[(ntopics+1):length(ω)]::Array{Float64,1}
    else
        ω = μ_ω*ones(ntopics+J)::Array{Float64,1}
        ω_post = zeros(ntopics+J+(ntopics*J), M_iteration)
        ω_x = ω[(ntopics+1):(ntopics+J)]::Array{Float64,1}
    end
    if length(ω_init) == length(ω)
        ω = ω_init
    end
    #ω_new = μ_ω*ones(ntopics+J+(ntopics*J))::Array{Float64,1}
    ω_new = μ_ω*ones(length(ω))::Array{Float64,1}
    ω_z = ω[1:ntopics]::Array{Float64,1}

    S_inv = (1/σ_ω).*Array{Float64,2}(I(length(ω)))
    Σ = Array{Float64,2}(I(length(ω)))
    σ2_post = zeros(M_iteration)


    # Extract the documents and initialise the topic assignments
    for dd in 1:D
        topic_base_document = Lda.TopicBasedDocument(ntopics)
        for wordid in 1:V
            for _ in 1:dtm_Estep[dd,wordid]
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
    if length(docs_init) == length(docs) && length(topics_init) == length(topics)
        docs = deepcopy(docs_init)
        topics = deepcopy(topics_init)
    end

    # Vector for topic probabilities
    probs::Array{Float64,1} = Vector{Float64}(undef, ntopics)
    # Vector of topic counts
    topic_counts::Array{Array{Int64,1},1} = getfield.(docs, :topicidcount)

    # Placeholders to store the document-topic and topic-word distributions
    θ::Array{Float64,2} = zeros(ntopics,D)
    θ_avg::Array{Float64,2} = zeros(ntopics,D)
    θ_Mstep::Array{Float64,2} = zeros(ntopics,size(y_Mstep,1))
    β::Array{Float64,2} = zeros(ntopics,V)
    β_avg::Array{Float64,2} = zeros(ntopics,V)
    term1_avg::Array{Float64,2} = zeros(ntopics,D)
    term2_avg::Array{Float64,2} = zeros(ntopics,D)
    term3_avg::Array{Float64,2} = zeros(ntopics,D)
    # Optional placeholder for the average Z in each document across iterations
    Z_bar::Array{Float64,2} = zeros(ntopics,D)
    Z_bar_avg::Array{Float64,2} = zeros(ntopics,D)
    Z_bar_Mstep::Array{Float64,2} = zeros(ntopics,size(y_Mstep,1))
    ω_zx::Array{Float64,1} = zeros(ntopics*J)
    ω_docspec::Array{Float64,1} = zeros(ntopics)::Array{Float64,1}
    inter_effects::Array{Float64,2} = zeros(size(y_Mstep,1),(ntopics*J))::Array{Float64,2}
    regressors::Array{Float64,2} = zeros(size(y_Mstep,1), length(ω))

    for em in 1:EM_iteration
        # Gibbs sampling E-step
        display(join(["Gibbs Sampling for E step ", em]))
        #@showprogress 1
        prog = Progress(E_iteration, 1)
        for it in 1:E_iteration
            # Run through all documents, assigning z
            for (dd,doc) in enumerate(docs)
                # Extract covariates for doc, if there are any
                if J > 0 && !interactions
                    x_dd = x_Estep[dd,:]::Array{Float64,1}
                    ω_docspec = ω_z
                elseif J > 0 && interactions
                    ω_docspec = zeros(ntopics)
                    for jj in 1:J
                        col_range = (jj*ntopics-ntopics+1):(jj*ntopics)
                        ω_docspec.+= ω_zx[col_range]*x_Estep[dd,jj]
                    end
                    ω_docspec .+=ω_z
                else
                    x_dd = [0.]::Array{Float64,1}
                    ω_docspec = ω_z
                end
                # Run through each word in doc
                for (nn, word) in enumerate(doc.text)
                    topicid_current = doc.topic[nn] # Current id of this word
                    doc.topicidcount[topicid_current] -= 1 # Remove this word from doc topic count counts
                    topics[topicid_current].count -= 1 # Remove from overall topic count
                    topics[topicid_current].wordcount[word] -= 1 # Remove from word-topic count
                    N_d = length(doc.text) # Document length (minus this word)

                    # Residual for y given all other topic assignments
                    if J > 0 && !interactions
                        res = (y_Estep[dd] - dot(ω_x, x_dd) - dot(ω_z,doc.topicidcount)./N_d)
                    elseif J > 0 && interactions
                        res = (y_Estep[dd] - dot(ω_docspec,doc.topicidcount)./N_d)
                    else
                        res = (y_Estep[dd] - dot(ω_docspec,doc.topicidcount)./N_d)
                    end
                    for kk in 1:ntopics
                        term1 = (doc.topicidcount[kk] + α) # prob of topic based on rest of doc
                        term2 = (get(topics[kk].wordcount, word, 0)+ η) / (topics[kk].count + η * V) # prob of word based on topic
                        term3 = exp((1/(2*σ2))*((2*ω_docspec[kk]/N_d)*res -  (ω_docspec[kk]/N_d)^2)) # predictive distribution for y
                        probs[kk] = term1 * term2 * term3

                        if it == E_iteration
                            term1_avg[kk,dd] += (1/N_d)*term1::Float64
                            term2_avg[kk,dd] += (1/N_d)*term2::Float64
                            term3_avg[kk,dd] += (1/N_d)*term3::Float64
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

            # Could add Gibbs sampler for ω and σ here for full Gibbs without EM

            ### Add contribution to average β, θ and Z_avg
            β_avg += (β.*(1/E_iteration))::Array{Float64,2}
            θ_avg += (θ.*(1/E_iteration))::Array{Float64,2}
            Z_bar_avg += (Z_bar.*(1/E_iteration))::Array{Float64,2}

            next!(prog)

        end # End of Gibbs sampled iterations
        # Renormalise θ and β to eliminate any numerical errors
        Z_bar_avg ./= sum(Z_bar_avg, dims=1)::Array{Float64,2}
        θ_avg ./= sum(θ_avg, dims=1)::Array{Float64,2}
        β_avg ./= sum(β_avg, dims=2)::Array{Float64,2}

        @assert any(Z_bar_avg.!= 1/ntopics) "Need some documents in each step"

        # M step (Z first, then x)
        if batch
            display(join(["M step topic assignments"]))
            θ_Mstep, Z_bar_Mstep =
                LDA_Gibbs_predict(dtm_Mstep, ntopics, β_avg,
                α = α, iteration = M_iteration, burnin = 100)
            @assert any(Z_bar_Mstep.!= 1/ntopics) "Need some documents in each step"
        else
            Z_bar_Mstep= Z_bar_avg
        end
        #
        if J > 0
            if !interactions
                regressors = hcat(transpose(Z_bar_Mstep), x_Mstep)::Array{Float64,2}
            else
                for jj in 1:J
                    col_range= (jj*ntopics-ntopics+1):(jj*ntopics)
                    inter_effects[:,col_range] = transpose(Z_bar_Mstep).*vec(x_Mstep[:,jj])
                end
                regressors = hcat(transpose(Z_bar_Mstep), inter_effects)::Array{Float64,2}
            end
        else
            regressors = hcat(transpose(Z_bar_Mstep))::Array{Float64,2}
        end

        # If no or improper prior for sigma given, then just iterate until σ2 converges
        if a_0 <= 0. && b_0 <= 0.
            for _ in 1:M_iteration

                Σ = inv(σ2^(-1)*(transpose(regressors)*regressors) .+ S_inv)::Array{Float64,2}
                ω_new = (σ2^(-1)*Σ*(transpose(regressors)*y_Mstep))::Array{Float64,1}
                if est_σ2
                    σ2_new = mean((y_Mstep .- regressors*ω_new).^2)
                else
                    σ2_new = σ2
                end
                if abs(σ2_new - σ2) < 0.001
                    break
                else
                    σ2 = σ2_new
                end
            end
        else
            display(join(["Gibbs Sampling M-step: ", em]))
            ω_post, σ2_post = BLR_Gibbs(y_Mstep, regressors, iteration = M_iteration,
                m_0 = μ_ω, V_0 = σ_ω, a_0 = a_0, b_0 = b_0)
            ω_new = Array{Float64,1}(vec(mean(ω_post, dims = 2)))
            σ2 = mean(σ2_post)
        end

        if maximum(abs.(ω .- ω_new)) < ω_tol
            ω = ω_new::Array{Float64,1}
            ω_z = ω[1:ntopics]::Array{Float64,1}
            if !interactions
                ω_x = ω[(ntopics+1):(ntopics+J)]::Array{Float64,1}
            else
                ω_zx = ω[(ntopics+1):length(ω)]::Array{Float64,1}
            end
            break
        else
            ω = ω_new::Array{Float64,1}
            ω_z = ω[1:ntopics]::Array{Float64,1}
            if !interactions
                ω_x = ω[(ntopics+1):(ntopics+J)]::Array{Float64,1}
            else
                ω_zx = ω[(ntopics+1):length(ω)]::Array{Float64,1}
            end
        end

        # Extract coefficients for z and x
        #ω_z = ω[1:ntopics]::Array{Float64,1}
        #ω_x = ω[(ntopics+1):(ntopics+J)]::Array{Float64,1}

        display(join(["M step complete, MSE: ", σ2]))
        display(join(["Coefficients:", ω], " "))


    end
    return (β = β_avg, θ = θ_avg, Z_bar = Z_bar_avg,
        ω = ω, Σ = Σ, σ2 = σ2, docs = docs, topics = topics,
        ω_post = ω_post, σ2_post = σ2_post, Z_bar_Mstep = Z_bar_Mstep)

end
