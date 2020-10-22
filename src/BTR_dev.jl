using SparseArrays, LinearAlgebra, CSV, KernelDensity, Parameters
using Random, Distributions, DataFrames, GLM, StatsBase, JSON
using TextAnalysis, Plots, ProgressMeter, ColorSchemes, Plots.PlotMeasures


# Structures
include("BTR_structs.jl")

# Import various functions
include("BTR_preprocessing.jl")
include("BTR_aux_functions.jl")
include("BTR_Estep.jl")
include("BTR_Mstep.jl")
include("BTR_EMGibbs.jl")
include("BTRpredict.jl")
include("BTR_visualisation.jl")
include("BTRfullGibbs.jl")

include("BTR_EMGibbs_paras.jl")
include("BTR_Gibbs_predict.jl")
include("LDAGibbs.jl")
include("LDA_Gibbs_predict.jl")
include("BTR_Gibbs_predict_paras.jl")
include("BTR_sentiment.jl")

foo(x::T, y::T) where T <: Real = x + y - 5
bar(z::Float64) = foo(sqrt(z), z)
greet() = print("Hello World!")
