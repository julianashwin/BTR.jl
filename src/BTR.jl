using SparseArrays, LinearAlgebra, CSV,KernelDensity
using Random, Distributions, DataFrames, GLM, StatsBase
using TextAnalysis, Plots, ProgressMeter, ColorSchemes



module Lda

mutable struct TopicBasedDocument
    topic::Vector{Int}
    text::Vector{Int}
    topicidcount::Vector{Int}
end
TopicBasedDocument(ntopics) = TopicBasedDocument(Vector{Int}(), Vector{Int}(), zeros(Int, ntopics))

mutable struct TopicBasedParagraphDocument
    topic::Vector{Int}
    paragraphs::Vector{TopicBasedDocument}
    topicidcount::Vector{Int}
end
TopicBasedParagraphDocument(ntopics) = TopicBasedDocument(Vector{Int}(), Vector{TopicBasedDocument}(), zeros(Int, ntopics))


mutable struct Topic
    count::Int
    wordcount::Dict{Int, Int}
end
Topic() = Topic(0, Dict{Int, Int}())

end

include("BTR_aux_functions.jl")
include("BTR_Gibbs.jl")
include("BTR_EMGibbs.jl")
include("BTR_EMGibbs_paras.jl")
include("BTR_Gibbs_predict.jl")
include("LDA_Gibbs.jl")
include("LDA_Gibbs_predict.jl")
include("BTR_Gibbs_predict_paras.jl")
include("BTR_sentiment.jl")
