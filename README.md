# BTR.jl
A Julia package to estimate Bayesian Topic Regression (BTR) models, as described in Ahrens et al (2021).

## Installion

To install the package, first enter package mode in julia by entering "]". The package can then be installed either directly from GitHub with 
```julia
pkg> dev https://github.com/julianashwin/BTR.jl
```
or if the repository has been downloaded, from the path to the BTR.jl folder (note, this is the path to the BTR.jl *folder*, not to the file of the same name in the src folder).
```julia
pkg> dev path_to_folder/BTR.jl
```
Once the pakage mode is exited (cmd+c), the package can then be loaded with 
```julia
using BTR
```

## Example files
This repository contains several example files that demonstrate how to use the package to estimate BTR and other topic models.
* `BTR_synthetic.jl` runs through an example in which the synthetic dataset used in the paper is generated, prepared for estimation and then a variety of models estimated on it.
* `BTR_semisynthetic_Booking.jl` and `BTR_semisynthetic_Yelp.jl` generates the data and estimates treatment effects for the semi-synthetic examples in the paper.
* `BTR_Booking.jl` and `BTR_Yelp.jl` evaluates out of sample performance of BTR and other sampling-based topic models on customer review datasets.
* `BTC_synthetic.jl` runs through an example for classification rather than regression with a synthetic dataset. 
Code for the rSCHOLAR model that is also used as a benchmark can be found in this [repository](https://github.com/MaximilianAhrens/scholar4regression).



## Pre-processing

Before estimating we need to convert the text-data into a document-term-matrix (DTM) for which we can use the DocumentTermMatrix function from the [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) package, and convert the numerical data into appropriate arrays. To ensure that the raw data is in the correct format and to facilitate train-test set division, it is recommended to use the BTRRawData structure as described below

### Text data

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
As it is poossible to have multiple documents/paragraphs per observation, we also need to define the document id for each row of the DTM. This document id variable needs to be an array of integers. If the there is one document per observation, this can simply be row ids for the DTM.
```julia
docidx_dtm = Array{Int64,1}(1:length(text_clean))
```
If you wish to split the sample into training and test sets, this should be done after pre-processing to ensure that the DTM vocabulary is consistent across the two sets.

### Numerical data

The response variable `y` and the non-text regression features `x` should be formated as a one and two dimensional array of floats respectively. 
```julia
x = Array{Float64,2}(hcat(df.sentiment,df.stars_av_u, df.stars_av_b))
y = Array{Float64,1}(df.stars)
```
As with the text data, we need to define a document id for these numerical variables as an array of integers.
```julia
docidx_vars = df.doc_idx
```

### BTRRawData and training-test splitting

There are two ways to convert data into a BTRRawData structure
1. Explicitly define from the DTM, id variables, y, vocab and optionally covariates x
2. Using the `btr_traintestsplit` function


## Options

A BTROptions object keeps track of the modelling and estimation options, each field has a default as described below. These same options can also be used to estimate unsupervised LDA models, and an equivalent object exists for classification rather than regression. This options object can be initialised at default values with
```julia
opts = BTROptions()
```
The individual can then be edited individually, for example
```julia
opts.ntopics = 20
opts.xregs = [1,2,3]
```
Alternatively, fields can be set when defining the object
```julia
opts = BTROptions(ntopics = 20, xregs = [1,2,3])
```
Fields and default values for BTROptions are:
* `ntopics::Int64 = 2`: number of topics (K)
* `interactions::Array{Int64,1} = Array{Int64,1}([])`: which x variables should be interacted with topic proportions
* `xregs::Array{Int64,1} = Array{Int64,1}([])`: which x variables to be included in the regression
* `emptydocs::Symbol = :prior`: how to deal with empty documents, topic proportions can be set to the Dirichlet prior or zero
* `interlevel::Symbol = :doc`: level at which interactions between x and topic proportions are calculated, can be at the document or paragraph level
* `E_iters::Int64 = 100`: Number of sampling iterations in E-step that approximates the distribution topic proportions
* `M_iters::Int64 = 250`: Number of sampling iterations in M-step that approximates the distribution of regression parameters
* `EM_iters::Int64 = 10`: Maximum number of E-M steps (will stop earlier if convergence reached)
* `fullGibbs_iters::Int64 = 10000`: Number of iterations under standard Gibbs sampling (without EM), particularly relevant for unsupervised LDA
* `fullGibbs_thinning::Int64 = 10`: Thinning interval for Gibbs sampling
* `burnin::Int64 = 10`: Burnin iterations discarded at the beginning of any sampling
* `mse_conv::Int64 = 0`: Toggle whether M-step MSE is used to assess convergence (if zero then only regression parameters ω used)
* `ω_tol::Float64 = 0.01`: Convergence tolerance for regression parameters ω
* `rel_tol::Bool = false;`: Toggle whether convergence tolerance in ω is scaled relative to the size of ω
* `CVEM::Symbol = :none`: Toggle whether CVEM approach is used; options are :none for no CVEM, :obs to split into E and M step samples at the observation level and :paras to split at the paragraph level
* `CVEM_split::Float64 = 0.75`: Proportion of observations/paragraphs used in the E-step (to estimate topic proportions) with the remainder used in the M-step (to estimate regression parameters
* `α::Float64 = 1.`: Dirichlet prior on document-topic distribution
* `η::Float64 = 1.`: Dirichlet prior on topic-vocabulary distribution
* `σ_ω::Float64 = 1.`: Variance of Gaussian prior on regression coefficients
* `μ_ω::Float64 = 0.`: Mean of Gaussian prior on regression coefficients
* `a_0::Float64 = 0.`: Shape of Inverse-Gamma prior on residual variance
* `b_0::Float64 = 0.`: Scale of Inverse-Gamma prior on residual variance
* `plot_ω = Bool(true)`: Toggle whether to plot progress of mse and ω coefficients between E-M steps



## Converting BTRRawData into a BTRCorpus object






## Estimation

The core function for estimation is `BTR_EMGibbs`, which estimates a BTR model with an EM-Gibbs algorithm.
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
* `dtm_in::SparseMatrixCSC{Int64,Int64}`: the DTM, as a sparse matrix.
* `ntopics::Int64`: the number of topics you want the model to have, as an integer.
* `y::Array{Float64,1}`: the response variable, as a one-dimensional array of floats.

There are then a number of optional key-word arguments:
* `x::Array{Float64,2}`: the non-text regression features, the default is to run a BTR without any additional regression features (i.e. a supervised LDA).
* `σ2::Float64`: if specified, the residual variance `σ2` is not estimated but taken as known, if not specified it will be estimated.
* `α::Float64`: the Dirichlet prior on document-topic proportions `θ`, default is `1.0`.
* `η::Float64`: the Dirichlet prior on topic-vocabulary proportions `β`, default is `1.0`.
* `σ_ω::Float64`: variance of Gaussian prior on regression coefficients `ω`, default is `1.0`.
* `μ_ω::Float64`: mean of Gaussian prior on regression coefficients `ω`, default is `0.0`.
* `a_0::Float64`: shape of Inverse-Gamma prior on residual variance `σ2`, if not specified the model is estimated without prior on `σ2`.
* `b_0::Float64`: scale of Inverse-Gamma prior on residual variance `σ2`, if not specified the model is estimated without prior on `σ2`.
* `E_iteration::Int`: number of iterations per E-step, default is `100`.
* `M_iteration::Int`: number of iterations per M-step, default is `500`.
* `EM_iteration::Int`: maximum number of EM-iterations if convergence not reached, default is `10`.
* `burnin::Int`: number of iterations used as a burnin in Gibbs sampling, default is `10`.
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

The output of the function is a NamedTuple with the following elements:
* `β`: mean of posterior distribution of topic-vocabulary distribution from last E-step.
* `θ`: mean of posterior distribution of document-topic distribution from last E-step.
* `Z_bar`: mean topic assignments of posterior distribution from last E-step.
* `ω`:  mean of posterior distribution for regression coefficients from last M-step. 
* `Σ`: mean of posterior distribution for regression feature covariance matrix from last M-step (will only be calculated when there **is no** prior on `σ2`).
* `σ2`: mean of posterior distribution for residual variance.
* `docs`: document-topic assignments from the last iteration of the last E-step. 
* `topics`: topic-vocabulary assignments from the last iteration of the last E-step. 
* `ω_post`: sampled posterior distribution for regression coefficients from last M-step (will only be calculated when there **is** an Inverse-Gamma prior on `σ2`).
* `σ2_post`: sampled posterior distribution for residual variance from last M-step (will only be calculated when there **is** an Inverse-Gamma prior on `σ2`).
* `Z_bar_Mstep`: mean topic assignments of posterior distribution for the last M-step (useful is using Cross-Validation AM approach).
* `ω_iters`: evolution of across EM-iterations, which is used to assess convergence.

So if, for example, you want to estimate estimate a 10 topic BTR model with an `InverseGamma(4.0,4.0)` prior on `σ2` and topic-interaction terms on the second and third columns of `x`, with Cross-Validation EM, this would be:
```julia
results_btr = BTR_EMGibbs_paras(dtm_sparse, 10, y, x = x, a_0 = 4.0, b_0 = 4.0,
        interactions = [2,3], batch = true)
```
All other priors and estimation options would be set as per the default values described above.

  
## Prediction
  
The core function for prediction is `BTR_Gibbs_predict`, which produces predictions for the response variable, given some regression features and previously estimated topic-vocabulary distribution `β` and regression coefficients `ω`.
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
There are four necessary arguments:
* `dtm_in::SparseMatrixCSC{Int64,Int64}`: the DTM.
* `ntopics::Int`: the number of topics.
* `β::Array{Float64,2}`: the topic-vocabulary distribution to be used in prediction (having previously been estimated by, for example, `BTR_EMGibbs`).
* `ω::Array{Float64,1}`: the regression coefficients, note that the dimensionality must match that implied by `ntopics` and the optional arguments `x` and `interactions`. 

There are then also some additional optional arguments:
* `x::Array{Float64,2}`: additional non-text regression features, if not specified the function assumes that the only regression features are the topics.
* `Σ::Array{Float64,2}`: the covariance of the estimated regression features, to produce predictive distributions for `y` (not fully implemented yet).
* `σ2::Float64`: the residual variance, to produce predictive distributions for `y` (not fully implemented yet). 
* `y::Array{Float64,1}`: the true values of the response variables being predicted, if specified the MSE of the predictions is calculated. 
* `α::Float64 =1.`: Dirichlet prior on the document-topic distribution, default is `1.0`.
* `E_iteration::Int64`: number of iterations for Gibbs sampler estimating the topic assignments of the documents, default is `100`.
* `burnin::Int64`number of iterations used as a burnin in Gibbs sampling, default is `10`.
* `interactions::Array{Int64,1}`: if specified includes interactions between the topic features and specified columns of `x`, default is no interactions. This needs to match the dimensionality of `ω`.

The output of the function is a NamedTuple with the following elements:
* `y_pred`: point predictions for the response variables associated with the inputted regression features.
* `θ`: mean of posterior distribution for document-topic distribution of text regression features.
* `Z_bar`: mean topic assignments of posterior distribution of text regression features.
* `ω`: mean of posterior distribution for regression coefficients from last M-step.
* `mse_oos`: the MSE of predicted response variables (will be zero if `y` not included as a key-word argument.
* `docs`: document-topic assignments from the last iteration of Gibbs sampling. 
* `topics`: topic-vocabulary assignments from the last iteration of Gibbs sampling. 


## Multiple-paragraphs

The functions `BTR_EMGibbs_paras` and `BTR_Gibbs_predict_paras` are analogues of those above which estimate a BTR with multiple paragraphs/documents per observation. The key difference is that the `doc_idx::Array{Int64,1}` key-word argument can be used to associate each row of the DTM with a document. The model is then estimated with each paragraph (i.e. row of the DTM) having and independent `θ` distribution, but with the regression estimated at the document level (as encoded in `doc_idx`. This functionality will eventually be folded into the main `BTR_EMGibbs` and `BTR_Gibbs_predict` functions, but is kept separate for now.


## Benchmarking functions

There are functions which estimate several other functions that can be used to benchmark the performance of BTR.
* `LDA_Gibbs`: estimates a standard LDA by Gibbs sampling.
* `LDA_Gibbs_predict`: estimates the topic-document distribution `θ` and topic-assignments of documents `Z_bar`, given a topic-vocabulary distribution `β`.
* `BLR_Gibbs`: estimates a Bayesian Linear Regression with Normal-Inverse-Gamma priors by Gibbs sampling.
* `BTR_Gibbs`: estimate BTR with pure Gibbs sampling (i.e. without the EM algorithm). This is not ready yet.
* `wordlist_counts`: calculates word counts for each document given a wordlist and an DTM. Can be used in conjunction with `LM_dicts` and `HIV_dicts` to calculate sentiment scores using the [Loughran and McDonald (2011)](https://www3.nd.edu/~mcdonald/Word_Lists_files/Documentation/Documentation_LoughranMcDonald_MasterDictionary.pdf) and [Harvard Gerenal Inquirer](http://www.wjh.harvard.edu/~inquirer/homecat.htm) dictionaries. 
  
  
  
## Visualisation

The core function for visualisation is BTR_plot, which generates a box-plot illustrating the posterior distribution for the regression coefficients and the most heavily weighted words in the corresponding topic-vocabulary distribution.
```julia
function BTR_plot(β::Array{Float64,2}, ω_post::Array{Float64,2};
    nwords::Int64 = 10, plt_title::String = "", left_mar::Int64 = 7,
    fontsize::Int64 = 6, title_size::Int64 = 14,
    interactions::Array{Int64,1}=Array{Int64,1}([]))
``` 
The image below illustrates this for an example using Yelp reviews. We can see that some topics are generally associated with more positive reviews (e.g. topic 8 which puts a high weight on terms like "time", "recommend" and "amaz"), while other topics are generally associated with more negative reviews (e.g. topic 1 which puts a high weight on terms like "wait", "lobster" and "time". Note that this example also shows an attractive feature of using a topic modelling framework for text feature extraction: the term "time" has a positive meaning when it appears with "amaz", but a negative meaning when it appears with "wait". 

<img src="/figures/YP_BTRplot_example.png" alt="BTRplot example" title="Yelp example"/>

You can use the key-word arguments to add a title to the plot and control the number of terms and font size of the labels. It is also possible to include coefficients for one set of interaction terms using the interactions argument, which works in the same way as for `BTR_EMGibbs`.

There is a similar function, `synth_data_plot`, for the 3 topic synthetic example generated by the `synthetic_example.jl` file. Rather than label each topic with the most heavily weighted terms, this produces a heatmap. See the `synthetic_example.jl` and `synthetic_example.txt` files for more on this.


## Organisation of src folder

The files in the src folder is organised as follows:
* `BTR.jl` imports dependencies, defines the structures used throughout and runs the other files in the folder.
* `BTR_aux_functions.jl` defines several auxilliary functions, particularly those used to generate synthetic data and visualise results.
* `BTR_Gibbs.jl` defines the `BTR_Gibbs` function described above.
* `BTR_EMGibbs.jl` defines the `BTR_EMGibbs` function described above.
* `BTR_EMGibbs_paras.jl` defines the `BTR_EMGibbs_paras` function described above.
* `BTR_Gibbs_predict.jl` defines the `BTR_Gibbs_predict` function described above.
* `LDA_Gibbs.jl` defines the `LDA_Gibbs` function described above.
* `LDA_Gibbs_predict.jl` defines the `LDA_Gibbs_predict` function described above.
* `BTR_Gibbs_predict_paras.jl` defines the `BTR_Gibbs_predict_paras` function described above.
* `BTR_sentiment.jl` imports sentiment dictionaries from `LoughranMcDonald_lists.csv` and `HarvardIV_lists.csv`, and defines a function to compute word counts using these dictionaries.
* `LoughranMcDonald_lists.csv` includes sentiment dictionaries from [Loughran and McDonald (2011)](https://www3.nd.edu/~mcdonald/Word_Lists_files/Documentation/Documentation_LoughranMcDonald_MasterDictionary.pdf)
* `HarvardIV_lists.csv` includes sentiment dictionaries from [Harvard Gerenal Inquirer](http://www.wjh.harvard.edu/~inquirer/homecat.htm) dictionaries. 
