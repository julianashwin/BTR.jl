using SparseArrays, LinearAlgebra, CSV, KernelDensity, Parameters, GLM
using Random, Distributions, DataFrames, GLM, StatsBase, JSON
using TextAnalysis, Plots, ProgressMeter, ColorSchemes, Plots.PlotMeasures


# Structures
include("BTR_structs.jl")

# Import various functions
include("BTR_preprocessing.jl")
include("BTR_aux_functions.jl")
include("BTR_Estep.jl")
include("BTR_Mstep.jl")
include("BTRemGibbs.jl")
include("BTRpredict.jl")
include("BTR_visualisation.jl")
include("BTRfullGibbs.jl")
include("BTRfullGibbspredict.jl")
include("LDAGibbs.jl")
include("BTR_sentiment.jl")
