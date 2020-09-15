# BTR.jl
Estimate Bayesian Topic Regression models

## Importing necessary functions

In order to estimate BTR models and plot their output, you will first need to import the necessary functions. First, make sure you are in the correct working directory (i.e. the one which contains the src folder). Then run the line
```julia
include("src/BTR.jl")
```
which will load both the estimation and visualisation functions into your workspace. The BTR_synthetic.jl file runs through an example in which a synthetic dataset is generated, prepared for estimation and then a variety of models estimated on it. More explanation for this file can be found in the sythetic_example.txt file.

## Pre-processing

Before estimating we need to convert the text-data into a document-term-matrix (DTM) for which we can use the DocumentTermMatrix function from the [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) package.


## Estimation

The core function for estimation is BTR_EMGibbs
```julia
BTR_EMGibbs(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int,
    y::Array{Float64,1};  x::Array{Float64,2} = zeros(1,1),
    σ2::Float64 = 0.,
    α::Float64 =1., η::Float64 = 1., σ_ω::Float64 = 1., μ_ω::Float64 = 0.,
    a_0::Float64 = 0., b_0::Float64  = 0.,
    E_iteration::Int = 100, M_iteration::Int = 500, EM_iteration::Int = 10,
    burnin::Int = 10,
    topics_init::Array = Array{Main.Lda.Topic,1}(undef, 1),
    docs_init::Array = Array{Main.Lda.TopicBasedDocument,1}(undef, 1),
    ω_init::Array{Float64,1} = zeros(1), ω_tol::Float64 = 0.01, rel_tol::Bool = false,
    interactions::Array{Int64,1}=Array{Int64,1}([]), batch::Bool=false, EM_split::Float64 = 0.75,
    leave_one_topic_out::Bool=false, plot_ω::Bool = false)
```
* Takes as arguments a document-term-matrix encoded as a sparse matrix; the number of topics as an integer; and the response variable as an array of floats.
  
  
## Prediction
  
The core function for prediction is BTR_Gibbs_predict
```julia
function BTR_Gibbs_predict(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int,
    β::Array{Float64,2}, ω::Array{Float64,1};
    x::Array{Float64,2} = zeros(1,1),
    Σ::Array{Float64,2} = zeros(2,2),
    σ2::Float64 = 0., y::Array{Float64,1} = zeros(1),
    α::Float64 =1.,
    E_iteration::Int64 = 100, burnin::Int64 = 10,
    interactions::Array{Int64,1}=Array{Int64,1}([]))
```
  
  
  
## Visualisation

The core function for visualisation is BTR_plot
```julia
function BTR_plot(β::Array{Float64,2}, ω_post::Array{Float64,2};
    nwords::Int64 = 10, plt_title::String = "", left_mar::Int64 = 7,
    fontsize::Int64 = 6, title_size::Int64 = 14,
    interactions::Array{Int64,1}=Array{Int64,1}([]))
```    
