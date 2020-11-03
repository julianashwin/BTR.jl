"""
This file contains functions used to preprocess data for Bayesian Topic Regressions
"""

"""
Function to create saveable dataframes for dtm and corpus metadata
"""
function dtmtodfs(dtm_in::SparseMatrixCSC{Int64,Int64}, doc_idx::Array{Int64,1},
        vocab::Array{String,1}; docnames::Array{String,1} = [""], save_dir::String = "")
    # Convert DTM sparse matrix into dataframe
    Is, Js, Vs = findnz(dtm_in)
    dtm_df = DataFrame([:I => Is, :J => Js, :V => Vs])

    # Create dataframe with rowids and metadata that corresponds to the DTM
    rowids = 1:size(dtm_in,1)
    if !(length(docnames) == rowids)
        docnames = string.(rowids)
    end
    rowid_df = DataFrame(id=rowids, name=docnames, docidx=doc_idx)

    # Create dataframe with terms in the vocab corresponding to the DTM
    vocab_df = DataFrame(term = vocab, term_id = 1:size(dtm_in,2))

    if save_dir != ""
        CSV.write(join([save_dir,"/rowids.csv"]), rowid_df)
        CSV.write(join([save_dir,"/terms.csv"]), vocab_df)
        CSV.write(join([save_dir,"/dtm.csv"]), dtm_df)
    end
    return dtm_df, rowid_df, vocab_df
end



"""
Function to split data into test and training sets
"""
function btr_traintestsplit(dtm_in::SparseMatrixCSC{Int64,Int64}, docidx_dtm::Array{Int64,1},
        docidx_vars::Array{Int64,1}, y::Array{Float64,1};
        x::Array{Float64,2} = zeros(1,1),
        train_split::Float64 = 0.75, shuffle_obs::Bool = true)
    # Extract total number of documents/observations
    N::Int64 = length(unique(vcat(docidx_dtm, docidx_vars)))
    # Shuffle the *document* ids (or don't if shuffle=false and we want to split by original order)
    idx::Array{Int64,1} = 1:N
    if shuffle_obs
        idx = shuffle(idx)
    end
    # Separate indices into test and training
    train_obs::Array{Int64,1} = Array{Int64,1}(view(idx, 1:floor(Int, train_split*N)))
    test_obs::Array{Int64,1} = setdiff(idx, train_obs)
    # Identify the doc_idx of the documents assigned to each set
    train_docs::BitArray{1} = in.(docidx_dtm, [train_obs])
    test_docs::BitArray{1} = in.(docidx_dtm, [test_obs])
    # Identify the indices that will be assigned to train and test
    train_vars::BitArray{1} = in.(docidx_vars,[train_obs])
    test_vars::BitArray{1} = in.(docidx_vars,[test_obs])
    # Split the document indices into training and test sets
    docidx_dtm_train::Array{Int64,1} = docidx_dtm[train_docs]
    docidx_dtm_test::Array{Int64,1} = docidx_dtm[test_docs]
    docidx_vars_train::Array{Int64,1} = docidx_vars[train_vars]
    docidx_vars_test::Array{Int64,1} = docidx_vars[test_vars]
    # Split the DTM into training and test
    dtm_train::SparseMatrixCSC{Int64,Int64} = dtm_in[train_docs,:]
    dtm_test::SparseMatrixCSC{Int64,Int64} = dtm_in[test_docs,:]
    # Split y (and x) into training and test
    y_train::Array{Float64,1} = y[train_vars]
    y_test::Array{Float64,1} = y[test_vars]
    x_train::Array{Float64,2} = zeros(1,1)
    x_test::Array{Float64,2} = zeros(1,1)
    if size(x,1) == length(y)
        display("x variables defined at whole document level")
        x_train = x[train_vars,:]
        x_test = x[test_vars,:]
    elseif size(x,1) == size(dtm_in,1)
        display("x variables defined at the paragraph level")
        x_train = x[train_docs,:]
        x_test = x[test_docs,:]
    else
        display("No x variables provided")
    end

    train_data = DocStructs.BTRRawData(dtm_train, docidx_dtm_train, docidx_vars_train, y_train, x_train)
    test_data = DocStructs.BTRRawData(dtm_test, docidx_dtm_test, docidx_vars_test, y_test, x_test)
    return train_data, test_data

end



"""
Function that converts raw data into BTRdocs format
"""
function create_btrcrps(dtm_in::SparseMatrixCSC{Int64,Int64}, docidx_dtm::Array{Int64,1},
        docidx_vars::Array{Int64,1}, y::Array{Float64,1}, x::Array{Float64,2},
        ntopics::Int64)
    # Need doc_idx labels to be divorced from the indices themselves to fill the array
    docidx_labels::Array{Int64,1} = unique(vcat(docidx_dtm, docidx_vars))
    D::Int64 = length(unique(docidx_labels))
    V::Int64 = size(dtm_in,2)


    """
    First, create the topic object that will keep track of assignments at the corpus level
    """
    # Empty array to be filled with Topics
    topics = Array{DocStructs.Topic,1}(undef, ntopics)
    # Iteratively fill with empty topics
    for ii in 1:ntopics
        topics[ii] = DocStructs.Topic()
    end

    """
    Second, create the document objects that keep track of assignments at the document level
    """
    # Empty array to be filled with BTRParagraphDocuments
    btrdocs = Array{DocStructs.BTRParagraphDocument,1}(undef, D)
    # Iteratively will with random topics and paragraph-documents from dtm
    prog = Progress(D, 1)
    for dd in 1:D
        idx = docidx_labels[dd]
        dtm_paras = SparseMatrixCSC{Int64,Int64}(dtm_in[(docidx_dtm .== idx),:])
        P_dd = size(dtm_paras,1)
        x_dd = hcat(x[(docidx_vars .== idx),:])
        y_dd = y[(docidx_vars .== idx)][1]
        btr_document = DocStructs.BTRParagraphDocument(ntopics, y_dd, x_dd, P_dd, idx)
        topic_base_paras = Array{DocStructs.TopicBasedDocument,1}(undef, P_dd)
        for pp in 1:P_dd
            individual_para = DocStructs.TopicBasedDocument(ntopics)
            for wordid in 1:V
                for _ in 1:dtm_paras[pp,wordid]
                    topicid = rand(1:ntopics) # initial topic assignment
                    update_target_topic = topics[topicid] # select relevant topic
                    update_target_topic.count += 1 # add to total words in that topic
                    update_target_topic.wordcount[wordid] = get(update_target_topic.wordcount, wordid, 0) + 1 # add to count for that word
                    topics[topicid] = update_target_topic # update that topic
                    push!(individual_para.topic, topicid) # add topic to document
                    push!(individual_para.text, wordid) # add word to document
                    individual_para.topicidcount[topicid] =  get(individual_para.topicidcount, topicid, 0) + 1 # add topic to topic count for document
                    btr_document.topicidcount[topicid] = get(btr_document.topicidcount, topicid, 0) + 1 # add topic to topic count for document
                end
            end
            btr_document.paragraphs[pp] = individual_para
        end
        btrdocs[dd] = btr_document # Populate the document array
        next!(prog)
    end

    btrcrps = DocStructs.BTRCorpus(btrdocs, topics, docidx_labels, D, ntopics, V)

    return btrcrps

end
create_btrcrps(rawdata::DocStructs.BTRRawData, ntopics::Int64) =
    create_btrcrps(rawdata.dtm, rawdata.docidx_dtm, rawdata.docidx_vars, rawdata.y, rawdata.x, ntopics)


"""
Function to extract DocStructs.Topic objects from Array of DocStructs.BTRParagraphDocument
"""
function gettopics(btrdocs::Array{DocStructs.BTRParagraphDocument,1})
    # Get ntopics by looking at length of topicidcount in first document
    ntopics = length(btrdocs[1].topicidcount)
    all_paras = vcat(map(x -> x.paragraphs, btrdocs)...)
    # Empty array to be filled with Topics
    topics = Array{DocStructs.Topic,1}(undef, ntopics)
    for ii in 1:ntopics
        topics[ii] = DocStructs.Topic()
    end
    # Iteratively fill with empty topics
    for para in all_paras
        N_p = length(para.text)
        for nn in 1:N_p
            word = para.text[nn]
            assigned = para.topic[nn]
            update_topic = topics[assigned]
            update_topic.wordcount[word] = get(update_topic.wordcount, word, 0) + 1 # add to count for that word
            update_topic.count += 1
        end
    end
    return topics
end


"""
Convert row of a DTM to a string
"""
function dtmrowtotext(dtm_row::SparseVector{Int64,Int64}, vocab::Array{String,1})::String

    rels::SparseVector{Bool,Int64} = dtm_row .> 0
    row_rels::Array{Int64,1} = Array{Int64,1}(dtm_row[rels])
    vocab_rels::Array{String,1} = vocab[rels]
    text_out::String = join(repeat.(vocab_rels, row_rels))
    return text_out
end


"""
Convert DTM into an array of strings
"""
function dtmtotext(dtm_in::SparseMatrixCSC{Int64,Int64}, vocab::Array{String,1})::Array{String,1}
    vocab_new::Array{String,1} = vocab.*=" "

    text::Array{String,1} = Array{String,1}(undef,size(dtm_in,1))
    for ii in 1:size(dtm_in,1)
        text[ii] = dtmrowtotext(dtm_in[ii,:], vocab_new)
    end

    return text
end


"""
End of script
"""
