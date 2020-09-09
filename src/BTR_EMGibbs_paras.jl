"""
BTR_EMGibbs plus interaction effects between x variables and topics in the regression
    order is (all topics)*x_1 + (all_topics)*x_2 etc...
"""
function BTR_EMGibbs_paras(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int,
    y::Array{Float64,1};  x::Array{Float64,2} = zeros(1,1),
    doc_idx::Array{Int64,1} = [0],
    σ2::Float64 = 0.,
    α::Float64 =1., η::Float64 = 1., σ_ω::Float64 = 1., μ_ω::Float64 = 0.,
    a_0::Float64 = 0., b_0::Float64  = 0.,
    E_iteration::Int = 100, M_iteration::Int = 500, EM_iteration::Int = 10,
    burnin::Int = 10,
    topics_init::Array = Array{Main.Lda.Topic,1}(undef, 1),
    docs_init::Array = Array{Main.Lda.TopicBasedDocument,1}(undef, 1),
    ω_init::Array{Float64,1} = zeros(1), ω_tol::Float64 = 0.01, rel_tol::Bool = false,
    interactions::Array{Int64,1}=Array{Int64,1}([]), batch::Bool=false, EM_split::Float64 = 0.75,
    leave_one_topic_out::Bool=false, plot_ω::Bool = false)

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

    Ndp_Estep = sum(dtm_Estep, dims = 2)
    Ndp_Mstep = sum(dtm_Mstep, dims = 2)

    # Extract relevant information about paragraphs etc
    if length(doc_idx)==size(dtm_in,1)
        display("Estimating with paragraphs/sentences")
    else
        display("Estimating without paragraphs/sentences")
        doc_idx = Array(1:size(dtm_in,1))
    end
    doc_idx_Estep = doc_idx[E_idx]
    doc_idx_Mstep = doc_idx[M_idx]


    # Arrays with parargraph ids for each document and word/paragraphs counts
    # The indexes where correspond to the doc_idx, not the dtm row
    doc_para_array = Array{Array{Int64,1},1}(undef, maximum(doc_idx))
    Ndp_array = Array{Array{Int64,1},1}(undef, maximum(doc_idx))
    Pd_array = Array{Int64,1}(undef, maximum(doc_idx))
    Nd_array = Array{Int64,1}(undef, maximum(doc_idx))
    for dd in 1:maximum(doc_idx)
        doc_para_array[dd] = findall(x->x==dd, doc_idx)
        paras = dtm_in[doc_para_array[dd],:]
        Ndp_array[dd] = vec(sum(paras,dims=2))
        Pd_array[dd] = length(Ndp_array[dd])
        Nd_array[dd] = sum(Ndp_array[dd])
    end
    #display(Nd_array)



    # Initialise variables
    DP, V = size(dtm_Estep)
    D_Estep = length(unique(doc_idx_Estep))
    D_Mstep = length(unique(doc_idx_Mstep))
    docs::Array{Lda.TopicBasedDocument,1} = Array{Lda.TopicBasedDocument,1}(undef, DP)
    topics::Array{Lda.Topic,1} = Array{Lda.Topic,1}(undef, ntopics)
    for i in 1:ntopics
        topics[i] = Lda.Topic()
    end


    # Deal with no x case and pre-specified σ2
    if size(x_Estep,1) == DP
        J = size(x_Estep,2)
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
    if σ2 == 0.
        est_σ2 = true
        σ2 = var(y_Estep)
    else
        est_σ2 = false
    end

    ## Regression variables
    # Are we leaving one topic out?
    if leave_one_topic_out
        n_ωz = ntopics - 1::Int64
    else
        n_ωz = ntopics::Int64
    end
    #ω = μ_ω*ones(ntopics+J+(ntopics*J))::Array{Float64,1}
    if length(interactions) > 0
        ω = μ_ω*ones(ntopics+J_inter + J_nointer)::Array{Float64,1}
        ω_post = zeros(length(ω), M_iteration)::Array{Float64,2}
        ω_zx = ω[(ntopics+1):(ntopics+J_inter)]::Array{Float64,1}
        ω_x = ω[(ntopics+J_inter+1):length(ω)]::Array{Float64,1}
    else
        ω = μ_ω*ones(ntopics+J)::Array{Float64,1}
        ω_post = zeros(ntopics+J+(ntopics*J), M_iteration)::Array{Float64,2}
        ω_x = ω[(ntopics+1):(ntopics+J)]::Array{Float64,1}
        ω_zx = [0.]::Array{Float64,1}
    end
    if length(ω_x) == 0
        ω_x = [0.]
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
    for dd in 1:DP
        topic_base_document = Lda.TopicBasedDocument(ntopics)
        for wordid in 1:V
            for _ in 1:dtm_Estep[dd,wordid]
                topicid = rand(1:ntopics) # initial topic assignment
                update_target_topic = topics[topicid] # select relevant topic
                update_target_topic.count += 1 # add to total words in that topic
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

    # Some temporary variables that are used in sampling
    probs = Vector{Float64}(undef, ntopics)::Array{Float64,1}
    x_dd = zeros(1,max(J_nointer,1))::Array{Float64,2}
    inter_d_minusp = zeros(1,max(1,J_inter))::Array{Float64,2}
    # Vector of topic counts
    topic_counts = getfield.(docs, :topicidcount)::Array{Array{Int64,1},1}

    # Placeholders to store the document-topic and topic-word distributions
    θ = zeros(ntopics,DP)::Array{Float64,2}
    θ_avg = zeros(ntopics,DP)::Array{Float64,2}
    θ_doc_avg = zeros(ntopics,D_Mstep)::Array{Float64,2}
    β = zeros(ntopics,V)::Array{Float64,2}
    β_avg = zeros(ntopics,V)::Array{Float64,2}

    # Placeholders for the Z variables at doc and para level for each step
    Z_bar = zeros(ntopics,DP)::Array{Float64,2}
    Z_bar_avg = zeros(ntopics,DP)::Array{Float64,2}
    Z_bar_Mstep = zeros(ntopics,size(y_Mstep,1))::Array{Float64,2}
    Z_Mstep = zeros(size(y_Mstep,1),ntopics)
    Z_bar_docs = zeros(D_Mstep, ntopics)::Array{Float64,2}
    #ω_zx = zeros(ntopics*J)::Array{Float64,1}
    ω_docspec = zeros(ntopics)::Array{Float64,1}
    inter_effects = zeros(length(y_Mstep),J_inter)::Array{Float64,2}
    inter_effects_docs = zeros(D_Mstep,J_inter)::Array{Float64,2}
    regressors = zeros(D_Mstep, length(ω))::Array{Float64,2}

    # Document-level x and y won't change throughout estimation
    if J_nointer > 0
        x_Mstep_docs = group_mean(x_Mstep[:,no_interactions],doc_idx_Mstep)
    else
        x_Mstep_docs = zeros(D_Mstep, 0)
    end
    y_Mstep_docs = vec(group_mean(hcat(y_Mstep),doc_idx_Mstep))

    # Placeholder to keep track of ω across EM iterations
    ω_iters = zeros(length(ω),EM_iteration+1)
    ω_diff = zeros(length(ω))::Array{Float64,1}

    ## Start estimation
    for em in 1:EM_iteration
        # Gibbs sampling E-step
        display(join(["Gibbs Sampling for E step ", em]))
        #@showprogress 1
        prog = Progress(E_iteration, 1)
        for it in 1:E_iteration
            # Run through all documents, assigning z
            for (dd,doc) in enumerate(docs)

                ## First up gather necessary metadata from other paras, x and y
                # Always need y_dd
                y_dd = y_Estep[dd]
                # Always need doc_id and N_s
                doc_id = doc_idx_Estep[dd]::Int64
                N_d = max(Nd_array[doc_id],1)::Int64
                # Only need x_dd if J_nointer > 0
                if J_nointer > 0
                    x_dd = reshape(x_Estep[dd,no_interactions],(1,J_nointer))::Array{Float64,2}
                end
                # Only need other paras if there are any
                para_ids = doc_para_array[doc_id]::Array{Int64,1}
                if length(para_ids) > 1
                    minusp_ids = filter(x -> x != dd, para_ids)
                    other_paras = docs[minusp_ids]::Array{Main.Lda.TopicBasedDocument,1}
                    if J > 0
                        other_x = x[minusp_ids,:]::Array{Float64,2}
                    end
                    n_d_minusp = Float64.(hcat(getfield.(other_paras, :topicidcount)...))

                    # Interactions if there are any
                    if J_inter > 0
                        inter_d_minusp = create_inter_effects(Matrix(n_d_minusp),
                            other_x[:,interactions],ntopics)
                    end
                else
                    # If no other paragraphs just set there to zero
                    n_d_minusp = zeros(1,ntopics)
                    inter_d_minusp = zeros(1,max(1,J_inter))::Array{Float64,2}
                end
                # calculate predicted y without this paragraph
                ypred_minusp = sum(x_dd*ω_x) + sum(inter_d_minusp*(ω_zx./N_d)) +
                        sum(transpose(n_d_minusp)*(ω_z./N_d))

                # Calculate the paragraph specific coefficients, taking interactions into account
                ω_docspec = zeros(ntopics)
                ω_docspec += ω_z
                if J_inter > 0
                    for jj in 1:length(interactions)
                        col_range = (jj*ntopics-ntopics+1):(jj*ntopics)
                        ω_docspec.+= ω_zx[col_range]*x_Estep[dd,interactions[jj]]
                    end
                end
                # Run through each word in doc
                for (nn, word) in enumerate(doc.text)
                    topicid_current = doc.topic[nn] # Current id of this word
                    doc.topicidcount[topicid_current] -= 1 # Remove this word from doc topic count counts
                    topics[topicid_current].count -= 1 # Remove from overall topic count
                    topics[topicid_current].wordcount[word] -= 1 # Remove from word-topic count
                    N_dp = length(doc.text) # Paragraph length

                    # Residual for y given all other topic assignments
                    res = (y_dd - ypred_minusp - dot(ω_docspec,doc.topicidcount)./N_d)
                    for kk in 1:ntopics
                        term1 = (doc.topicidcount[kk] + α) # prob of topic based on rest of doc
                        term2 = (get(topics[kk].wordcount, word, 0)+ η) / (topics[kk].count + η * V) # prob of word based on topic
                        term3 = exp((1/(2*σ2))*((2*ω_docspec[kk]/N_d)*res -  (ω_docspec[kk]/N_d)^2)) # predictive distribution for y
                        probs[kk] = term1 * term2 * term3

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
            Z_bar_Mstep =
                LDA_Gibbs_predict(dtm_Mstep, ntopics, β_avg,
                α = α, iteration = M_iteration, burnin = 100)
            #@assert !any(isnan.(Z_bar_Mstep)) "NaN in Z_bar_Mstep"
            #replace_nan!(Z_bar_Mstep, 1/ntopics)
            @assert any(Z_bar_Mstep.!= 1/ntopics) "Need some documents in each step"
        else
            Z_bar_Mstep= Z_bar_avg
        end

        ## Create regressors for M_step
        # Pre-multiply the Z_bars and interaction effects by Ndp for averaging across paragraphs
        Z_Mstep = max.(Ndp_Mstep,1).*transpose(Z_bar_Mstep)
        for jj in 1:length(interactions)
            col_range= (jj*ntopics-ntopics+1):(jj*ntopics)
            inter_effects[:,col_range] = Z_Mstep.*vec(x_Mstep[:,interactions[jj]])
        end
        # Divide
        Z_Mstep .*= (1 ./max.(Nd_array[doc_idx_Mstep],Pd_array[doc_idx_Mstep]))
        if length(interactions) > 0
            inter_effects .*= (1 ./max.(Nd_array[doc_idx_Mstep],Pd_array[doc_idx_Mstep]))
        end
        # Aggregate to doc level (sum Z_bar and inter_effects, mean for doc-level x and y)
        Z_bar_docs = group_sum(Z_Mstep,doc_idx_Mstep)
        if J_inter > 0
            inter_effects_docs = group_sum(inter_effects,doc_idx_Mstep)
        end

        regressors = hcat(Z_bar_docs, inter_effects_docs, x_Mstep_docs)

        # If no or improper prior for sigma given, then just iterate until σ2 converges
        if a_0 <= 0. && b_0 <= 0.
            for _ in 1:M_iteration

                Σ = inv(σ2^(-1)*(transpose(regressors)*regressors) .+ S_inv)::Array{Float64,2}
                ω_new = (σ2^(-1)*Σ*(transpose(regressors)*y_Mstep_docs))::Array{Float64,1}
                if est_σ2
                    σ2_new = mean((y_Mstep_docs .- regressors*ω_new).^2)
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
            ω_post, σ2_post = BLR_Gibbs(y_Mstep_docs, regressors, iteration = M_iteration,
                m_0 = μ_ω, V_0 = σ_ω, a_0 = a_0, b_0 = b_0)
            ω_new = Array{Float64,1}(vec(mean(ω_post, dims = 2)))
            σ2 = mean(σ2_post)
        end


        ω_iters[:,em+1]= ω_new

        # Calculate change in ω for convergence
        ω_diff = abs.(ω .- ω_new)
        ω_diff[ω_diff.<ω_tol] = zeros(sum(ω_diff.<ω_tol))
        if rel_tol
            ω_diff ./=abs.(ω)
        end

        display(join(["M step complete, MSE: ", σ2]))
        if plot_ω
            display(plot(transpose(ω_iters[:,1:(em+1)]),legend=false))
            display(join(["Coefficient updates:", ω_diff], " "))
        end

        if maximum(ω_diff) < ω_tol
            ω = ω_new::Array{Float64,1}
            ω_z = ω[1:ntopics]::Array{Float64,1}
            if J_nointer > 0
                    ω_x = ω[(ntopics+J_inter+1):length(ω)]::Array{Float64,1}
            end
            if J_inter > 0
                ω_zx = ω[(ntopics+1):(ntopics+J_inter)]::Array{Float64,1}
            end
            break
        else
            ω = ω_new::Array{Float64,1}
            ω_z = ω[1:ntopics]::Array{Float64,1}
            if J_nointer > 0
                    ω_x = ω[(ntopics+J_inter+1):length(ω)]::Array{Float64,1}
            end
            if J_inter > 0
                ω_zx = ω[(ntopics+1):(ntopics+J_inter)]::Array{Float64,1}
            end
        end


        # Extract coefficients for z and x
        #ω_z = ω[1:ntopics]::Array{Float64,1}
        #ω_x = ω[(ntopics+1):(ntopics+J)]::Array{Float64,1}




    end
    return (β = β_avg, θ = θ_avg, Z_bar = Z_bar_avg,
        ω = ω, Σ = Σ, σ2 = σ2, docs = docs, topics = topics,
        ω_post = ω_post, σ2_post = σ2_post, Z_bar_Mstep = Z_bar_Mstep,
        ω_iters = ω_iters)

end
