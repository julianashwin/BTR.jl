module BTR

using SparseArrays, LinearAlgebra, CSV, KernelDensity, Parameters, GLM
using Random, Distributions, DataFrames, GLM, StatsBase, JSON
using TextAnalysis, Plots, ProgressMeter, ColorSchemes, Plots.PlotMeasures

"""
Add GLM to dependencies, update the exports
"""


# Export structures
export DocStructs
export BTROptions, BTRModel, BTRPrediction
export BTCOptions, BTCModel, BTCPrediction
export AbstractDocument, Document
export FileDocument, StringDocument, TokenDocument, NGramDocument
export GenericDocument
export Corpus, DirectoryCorpus
export DocumentTermMatrix
# Export pre-processing functions
export dtmtodfs, gettopics
export btr_traintestsplit, create_btrcrps, btc_traintestsplit, create_btccrps
export dtmrowtotext, dtmtotext
# Auxilliary functions
export multinomial_draw, replace_nan!, create_inter_effects
export group_mean, group_sum, findfirst_group
export bartopics, generate_docs, BLR, BLR_Gibbs
# EM algorithm estimation functions
export BTRemGibbs, btrmodel_splitobs, btrmodel_combineobs
export BTCemGibbs, btcmodel_splitobs, btcmodel_combineobs
export BTREstep, BTRMstep, BTCEstep, BTCMstep
# Prediction functions
export BTRpredict, BTCpredict
# Visualisation function
export coef_plot, BTR_plot, plot_topics, synth_data_plot, synth_reorder_topics
# Other models and estimation strategies
export BTRfullGibbs, BTRfullGibbspredict
export LDAGibbs
# Sentiment analysis functions and lists
export wordlistcounts, LM_dicts, HIV_dicts






# Structures
include("BTR_structs.jl")
# Pre-processing functions
include("BTR_preprocessing.jl")
# Auxilliary functions
include("BTR_aux_functions.jl")
# EM-Gibbs functions
include("BTR_Estep.jl")
include("BTR_Mstep.jl")
include("BTRemGibbs.jl")
# Prediction functions
include("BTRpredict.jl")
# Visualisation functions
include("BTR_visualisation.jl")
# Other models/estimation strategies
include("BTRfullGibbs.jl")
include("BTRfullGibbspredict.jl")
include("LDAGibbs.jl")
# Sentiment lists and function
include("BTR_sentiment.jl")


greet() = print("Hello World!")

end # BTR module
