"""
This file defines the structures that are used in estimating Bayesian Topic Regressions
"""

"""
Document and topic structures used to store the data
"""
module DocStructs

using SparseArrays

# Structure including all the raw data to estimate BTR
mutable struct BTRRawData
    dtm::SparseMatrixCSC{Int64,Int64}
    doc_idx::Array{Int64,1}
    y::Array{Float64,1}
    x::Array{Float64,2}
    N::Int64
end
BTRRawData(dtm, doc_idx, y, x::Array{Float64,2}) = BTRRawData(dtm, doc_idx, y, x, length(y))
BTRRawData(dtm, doc_idx, y, x, N::Int64) = BTRRawData(dtm, doc_idx, y, x, N)
BTRRawData(dtm, doc_idx, y, N::Int64) = BTRRawData(dtm, doc_idx, y, zeros(1,1), N)

# Structure to keep track of assignments at the paragraph level
mutable struct TopicBasedDocument
    topic::Vector{Int}
    text::Vector{Int}
    topicidcount::Vector{Int}
end
TopicBasedDocument(ntopics) = TopicBasedDocument(Vector{Int}(), Vector{Int}(), zeros(Int, ntopics))

# Structure to keep track of assignments at the document level
mutable struct BTRParagraphDocument
    paragraphs::Vector{TopicBasedDocument}
    topicidcount::Vector{Int}
    y::Float64
    x::Array{Float64,2}
    idx::Int64
end
BTRParagraphDocument(ntopics, y, x) = BTRParagraphDocument(Vector{TopicBasedDocument}(),
    zeros(Int, ntopics), y, x, 0)
BTRParagraphDocument(ntopics, y, x, P, docidx) = BTRParagraphDocument(Vector{TopicBasedDocument}(undef,P),
    zeros(Int, ntopics), y, x, 0)
BTRParagraphDocument(ntopics, y, x, P, docidx) = BTRParagraphDocument(Vector{TopicBasedDocument}(undef,P),
    zeros(Int, ntopics), y, x, docidx)



# Structure to keep track of assignments at the corpus level
mutable struct Topic
    count::Int
    wordcount::Dict{Int, Int}
end
Topic() = Topic(0, Dict{Int, Int}())

end
