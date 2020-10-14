module BTR

using SparseArrays, LinearAlgebra, CSV, KernelDensity
using Random, Distributions, DataFrames, GLM, StatsBase, JSON
using TextAnalysis, Plots, ProgressMeter, ColorSchemes, Plots.PlotMeasures


export foo, bar
export Lda
export AbstractDocument, Document
export FileDocument, StringDocument, TokenDocument, NGramDocument
export GenericDocument
export Corpus, DirectoryCorpus
export DocumentTermMatrix
export BTR_EMGibbs, BTR_Gibbs, BTR_EMGibbs_paras
export BTR_Gibbs_predict, BTR_Gibbs_predict_paras
export LDA_Gibbs, LDA_Gibbs_predict
export wordlist_counts, LM_dicts, HIV_dicts
#export LM_dicts, HIV_dicts
# From aux_functions
export multinomial_draw, replace_nan!
export create_inter_effects, group_mean, group_sum, findfirst_group
export generate_docs, synth_reorder_topics
export coef_plot, synth_data_plot, BTR_plot, plot_topics
export BLR, BLR_Gibbs





# Lda structures used throughout
module Lda

mutable struct TopicBasedDocument
    topic::Vector{Int}
    text::Vector{Int}
    topicidcount::Vector{Int}
end
TopicBasedDocument(ntopics) = TopicBasedDocument(Vector{Int}(), Vector{Int}(), zeros(Int, ntopics))

mutable struct BTRParagraphDocument
    topic::Vector{Int}
    paragraphs::Vector{TopicBasedDocument}
    topicidcount::Vector{Int}
    y::Float64
    x::Array{Float64,2}
end
BTRParagraphDocument(ntopics, y, x) = BTRParagraphDocument(Vector{Int}(), Vector{TopicBasedDocument}(),
    zeros(Int, ntopics), y, x)


mutable struct Topic
    count::Int
    wordcount::Dict{Int, Int}
end
Topic() = Topic(0, Dict{Int, Int}())

end

# Structures
include("BTR_structs.jl")

# Import various functions
include("BTR_preprocessing.jl")
include("BTR_aux_functions.jl")
include("BTR_visualisation.jl")
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

end # BTR module
