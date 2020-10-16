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
function btr_traintestsplit(dtm_in::SparseMatrixCSC{Int64,Int64},
        doc_idx::Array{Int64,1}, y::Array{Float64,1};
        x::Array{Float64,2} = zeros(1,1),
        train_split::Float64 = 0.75, shuffle_obs::Bool = true)
    # Extract total number of documents/observations
    N::Int64 = length(unique(doc_idx))
    # Shuffle the *document* ids (or don't if shuffle=false and we want to split by original order)
    idx::Array{Int64,1} = 1:N
    if shuffle_obs
        idx = shuffle(idx)
    end
    # Separate indices into test and training
    train_idx::Array{Int64,1} = Array{Int64,1}(view(idx, 1:floor(Int, train_split*N)))
    test_idx::Array{Int64,1} = Array{Int64,1}(view(idx, (floor(Int, train_split*N+1):N)))
    # Identify the doc_idx of the documents assigned to each set
    train_docs::BitArray{1} = in.(doc_idx, [train_idx])
    test_docs::BitArray{1} = in.(doc_idx, [test_idx])
    # Split the document indices into training and test sets
    doc_idx_train::Array{Int64,1} = doc_idx[train_docs]
    doc_idx_test::Array{Int64,1} = doc_idx[test_docs]
    # Split the DTM into training and test
    dtm_train::SparseMatrixCSC{Int64,Int64} = dtm_sparse.dtm[train_docs,:]
    dtm_test::SparseMatrixCSC{Int64,Int64} = dtm_sparse.dtm[test_docs,:]
    # Split y (and x) into training and test
    y_train::Array{Float64,1} = y[train_idx]
    y_test::Array{Float64,1} = y[test_idx]
    x_train::Array{Float64,2} = zeros(1,1)
    x_test::Array{Float64,2} = zeros(1,1)
    if size(x,1) == length(y)
        x_train = x[train_idx,:]
        x_test = x[test_idx,:]
    else
        display("No x variables provided")
    end

    train_data = DocStructs.BTRRawData(dtm_train, doc_idx_train, y_train, x_train)
    test_data = DocStructs.BTRRawData(dtm_test, doc_idx_test, y_test, x_test)
    return train_data, test_data

end



"""
Function that converts raw data into BTRdocs format
"""
function create_btrdocs(dtm_in::SparseMatrixCSC{Int64,Int64},
        doc_idx::Array{Int64,1}, y::Array{Float64,1}, x::Array{Float64,2},
        ntopics::Int64)
    # Need doc_idx labels to be divorced from the indices themselves to fill the array
    docidx_labels = unique(doc_idx)
    D = length(unique(docidx_labels))
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
        dtm_paras = SparseMatrixCSC{Int64,Int64}(dtm_in[(doc_idx .== idx),:])
        P_dd = size(dtm_paras,1)
        x_dd = hcat(x[[dd],:])
        y_dd = y[dd]
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

    return btrdocs, topics, docidx_labels

end
create_btrdocs(rawdata::DocStructs.BTRRawData, ntopics::Int64) =
    create_btrdocs(rawdata.dtm, rawdata.doc_idx, rawdata.y, rawdata.x, ntopics)


"""
Function to extract DocStructs.Topic objects from
"""
function gettopics(btrdocs::Array{DocStructs.BTRParagraphDocument,1},ntopics::Int64)
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
End of script
"""
