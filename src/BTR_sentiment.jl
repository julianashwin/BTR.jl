include("LM_lists.jl")

LM_dicts = (
    Negative = Array(String.(skipmissing(LM_lists.Negative))),
    Positive = Array(String.(skipmissing(LM_lists.Positive))),
    Uncertainty = Array(String.(skipmissing(LM_lists.Uncertainty))),
    Litigious = Array(String.(skipmissing(LM_lists.Litigious))),
    StrongModal = Array(String.(skipmissing(LM_lists.StrongModal))),
    WeakModal = Array(String.(skipmissing(LM_lists.WeakModal))),
    Constraining = Array(String.(skipmissing(LM_lists.Constraining))))


include("HIV_lists.jl")

HIV_dicts = (
    Negative = Array(String.(skipmissing(unique(HIV_lists.Negative)))),
    Positive = Array(String.(skipmissing(unique(HIV_lists.Positive)))),
    Pleasure = Array(String.(skipmissing(unique(HIV_lists.Pleasure)))),
    Pain = Array(String.(skipmissing(unique(HIV_lists.Pain)))))


"""
Function to compute counts from word list using DTM
"""
function wordlistcounts(dtm_sparse::SparseMatrixCSC{Int64,Int64},vocab::Array{String,1},
        wordlist::Array{String,1})
    # Remove words from list that don't appear in corpus
    list_short::Array{String,1} = filter(x -> x in vocab,wordlist)
    # Identify which columns of the DTM we need to sum over
    list_index::BitArray{1} = in.(vocab, [list_short])
    # Count word frequencies per document
    word_score::Array{Int64,1} = vec(sum(Array(dtm_sparse[:,list_index]),dims=2))
    return word_score
end
