using SparseArrays, LinearAlgebra, CSV, KernelDensity
using Random, Distributions, DataFrames, GLM, StatsBase, JSON
using TextAnalysis, Plots, ProgressMeter, ColorSchemes, Plots.PlotMeasures



"""
Document and topic structures used to store the data
"""
module DocStructs

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

# Import various functions
include("BTR_aux_functions.jl")
include("BTR_Gibbs.jl")
include("BTR_EMGibbs.jl")
include("BTR_EMGibbs_paras.jl")
include("BTR_Gibbs_predict.jl")
include("LDA_Gibbs.jl")
include("LDA_Gibbs_predict.jl")
include("BTR_Gibbs_predict_paras.jl")
include("BTR_sentiment.jl")

foo(x::T, y::T) where T <: Real = x + y - 5
bar(z::Float64) = foo(sqrt(z), z)
greet() = print("Hello World!")
