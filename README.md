# BTR.jl
Estimate Bayesian Topic Regression models

## Importing necessary functions

In order to estimate BTR models and plot their output, you will first need to import the necessary functions. First, make sure you are in the correct working directory (i.e. the one which contains the src folder). Then run the line
```julia
include("src/BTR.jl")
```
which will load the necessary dependencies and functions into your workspace. The BTR_synthetic.jl file runs through an example in which a synthetic dataset is generated, prepared for estimation and then a variety of models estimated on it. More explanation for this file can be found in the sythetic_example.txt file.

## Pre-processing

Before estimating we need to convert the text-data into a document-term-matrix (DTM) for which we can use the DocumentTermMatrix function from the [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) package. 

To do this import the documents as an array of strings (stemming, stop-word removal etc. can be done separately or in Julia using the tools described in the [documentation](https://juliatext.github.io/TextAnalysis.jl/documents.html#Preprocessing-Documents-1) for TextAnalysis.jl). Then convert this array of strings into an array of StringDocuments
```julia
docs = StringDocument.(text_clean)
```
and convert this array into a Corpus.
```julia
crps = Corpus(docs)
update_lexicon!(crps)
```
This Corpus can then be converted into DTM and the vocab extracted (for visualisation later).
```julia
dtm = DocumentTermMatrix(crps)
vocab = dtm_sparse.terms
```
The estimation function takes the DTM as a `SparseMatrixCSC` object, which can be extracted from the DocumentTermMatrix.
```julia
dtm_sparse = dtm.dtm
```
The response variable `y` and the non-text regression features `x` should be formated as a one and two dimensional array of floats respectively. Optionally, you can also specify the `doc_idx` of each line in the DTM as an array on integers, if there are multiple paragraphs/documents per observation. The dimensionality and order of `x`,`y` and `doc_idx` must match that of `dtm_sparse`.

If you wish to split the sample into training and test sets, this should be done after pre-processing to ensure that the DTM vocabulary is consistent across the two sets.

## Estimation

The core function for estimation is BTR_EMGibbs
```julia
function BTR_EMGibbs(dtm_in::SparseMatrixCSC{Int64,Int64}, ntopics::Int,
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
There are three necessary arguments:
* `dtm_in::SparseMatrixCSC{Int64,Int64}`: the document-term-matrix.
* `ntopics::Int64`: the number of topics you want the model to have.
* `y::Array{Float64,1}`: the response variable.
There are then a number of optional key-word arguments:
* `x::Array{Float64,2}`: the non-text regression features, the default is to run a BTR without an additional regression features.
* `σ2::Float64`: if specified, the residual variance `σ2` is not estimated but taken as known, if not specified it will be estimated.
* `α::Float64`: the Dirichlet prior on document-topic proportions `θ`, default is `1.0`.
* `η::Float64`: the Dirichlet prior on topic-vocabulary proportions `β`, default is `1.0`.
* `σ_ω::Float64`: variance of Gaussian prior on regression coefficients `ω`, default is `1.0`.
* `μ_ω::Float64`: mean of Gaussian prior on regression coefficients `ω`, default is `0.0`.
* `a_0::Float64`: shape of Inverse-Gamma prior on residual variance `σ2`, if not specified the model is estimated without prior on `σ2`.
* `b_0::Float64`: scale of Inverse-Gamma prior on residual variance `σ2`, if not specified the model is estimated without prior on `σ2`.
* `E_iteration::Int`: number of iterations per E-step, default is 100.
* `M_iteration::Int`: number of iterations per M-step, default is 500.
* `EM_iteration::Int`: maximum number of EM-iterations if convergence not reached, default is 10.
* `burnin::Int`: number of iterations used as a burnin in Gibbs sampling, default is 10.
* `topics_init::Array`: initialised topics as an array of Lda.Topic objects, default is random assignment.
* `docs_init::Array`: initialised documents as an array of Lda.TopicBasedDocument objects, default is random assignment.
* `ω_init::Array{Float64,1}`: initialised regression coefficients `ω`, default is `μ_ω`.
* `ω_tol::Float64`: convergence tolerance for `ω`, default is `0.01`.
* `rel_tol::Bool`: toggles whether to use relative as well as abosulte convergence tolerance for `ω`, default is`false`.
* `interactions::Array{Int64,1}`: if specified includes interactions between the topic features and specified columns of `x`, default is no interactions.
* `batch::Bool`: toggles whether to split training set across E and M steps for Cross-Validation EM as in [Shinozaki and Ostendorf (2007)](https://ieeexplore.ieee.org/abstract/document/4218131?casa_token=IwLvmvoMOYIAAAAA:Jcxp0O3NWH0UBQ4ettMhZ9UviCWJ9JtqhnIVciwNYcxVA_HEJYQS9y4lLqgwPnlb9yS0EDU), default is `false`.
* `EM_split::Float64`: proportion of sample used in E vs M step, default is `0.75`.
* `leave_one_topic_out::Bool`: toggles whether to leave one topic out of the regression (not fully implemented yet), default is `false`.
* `plot_ω::Bool`: toggles whether to plot the evolution of `ω` across EM-iterations, default is `false`.
  
  
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


## Organisation of src code

The code in the src folder is organised as follows:
* BTR.jl imports dependencies, defines the structures used throughout and runs the other files in the folder.
* BTR_aux_functions.jl defines auxilliary functions, particularly those used to generate synthetic data and visualise results.
