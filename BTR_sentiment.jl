LM_lists = CSV.read("src/LoughranMcDonald_lists.csv")

LM_dicts = (
    Negative = Array(String.(skipmissing(LM_lists.Negative))),
    Positive = Array(String.(skipmissing(LM_lists.Positive))),
    Uncertainty = Array(String.(skipmissing(LM_lists.Uncertainty))),
    Litigious = Array(String.(skipmissing(LM_lists.Litigious))),
    StrongModal = Array(String.(skipmissing(LM_lists.StrongModal))),
    WeakModal = Array(String.(skipmissing(LM_lists.WeakModal))),
    Constraining = Array(String.(skipmissing(LM_lists.Constraining))))


HIV_lists = CSV.read("src/HarvardIV_lists.csv")

HIV_dicts = (
    Negative = Array(String.(skipmissing(unique(HIV_lists.Negative)))),
    Positive = Array(String.(skipmissing(unique(HIV_lists.Positive)))),
    Pleasure = Array(String.(skipmissing(unique(HIV_lists.Pleasure)))),
    Pain = Array(String.(skipmissing(unique(HIV_lists.Pain)))))

"""
Function that gets word counts from given list in DTM
"""
function wordlist_counts(dtm_sparse::DocumentTermMatrix,vocab::Array{String,1},list::Array{String,1})
    # Convert DTM to dataframe
    dtm_df = DataFrame(dtm(dtm_sparse, :dense))
    rename!(dtm_df, Symbol.(vocab))

    # Remove words from list that don't appear in corpus
    list_short = filter(x -> x in vocab,list)

    # Count word frequencies per document
    word_score = sum(Array(dtm_df[:,Symbol.(list_short)]),dims=2)

    return word_score
end
