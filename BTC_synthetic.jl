"""
Implementation of Bayesian Topic Classification on a synthetic dataset
"""
## Set working directory (needs to have the src folder in it)
cd("/Users/julianashwin/Documents/GitHub/BTR.jl/")

"""
To make sure the latest version of the package is used run
pkg> dev /Users/julianashwin/Documents/GitHub/BTR.jl
or
pkg> dev https://github.com/julianashwin/BTR.jl

"""

using Revise, BTR
#include("src/BTR_dev.jl")
using TextAnalysis, DataFrames, CSV, Plots, GLM, LinearAlgebra
using StatsPlots, StatsBase, Plots.PlotMeasures, Distributions, Random


"""
Plotting options
"""
## Set the plotting backend: gr() is faster, pyplot() is prettier
gr()
#pyplot()
#plotly()
save_files = false # Toggle whether you want to save figures and data as files


"""
Generate some synthetic data
"""
## Size of sample and number of topics
K = 3 # number of topics
V = 9 # length of vocabulary (number of unique words)
D = 50000 # number of documents
Pd = 4 # number of paragraphs
Np = 25 # number of words per document
N = D+10 # total observations (N>=D)
NP = N*Pd # total number of paragraphs
DP = D*Pd # total number of non-empty paragraphs

## Parameters of true regression model
ω_z_true = [-1.,0.,1.] # coefficients on topics
#ω_zx_true = [2.,0.,1.] # coefficients on (topic,x) interactions
ω_x_true = [1., 0.] # coefficients on (topic,x) interactions
#ω_true = cat(ω_z_true, ω_zx_true, ω_x_true, dims = 1) # all coefficients
ω_true = cat(ω_z_true, ω_x_true, dims = 1) # all coefficients


## Dirichlet priors for topics
α = 1.0 # prior for θ (document-topic distribution)
η = 1.0 # prior for β (topic-word distribution)

## Draw latent variables θ and β
θ_true = rand(Dirichlet(K,α),DP)
θ_true = θ_true[:, sortperm(θ_true[1,:])]
θ_all = hcat(θ_true, (1/K).*ones(K,(NP-DP)))
β_true = bartopics(η, K, V)

heatmap(β_true, title = "", xlabel = "Vocab", ylabel = "Topic", yticks = 1:K,
    left_margin = 3mm,top_margin = 3mm, bottom_margin = 0mm)
if save_files; plot!(size =(200,300)); savefig("figures/synth_BTC/synth_class_true_beta.pdf"); end;

## Generate string documents and topic assignments
docs, Z_true, topic_counts, word_counts = generate_docs(DP, Np, K, θ_true, β_true)
docs_all = repeat([""], NP)
docs_all[1:DP] = docs

## Generate document indicator for each document
docidx_dtm = repeat(1:N,inner=Pd)
docidx_vars = Array{Int64,1}(1:N)

## Generate topic assignment regressors
Z_bar_true = Float64.(topic_counts)
Z_bar_true ./= sum(Z_bar_true, dims = 2)
Z_bar_all = (1/K)./ones(NP,K)
Z_bar_all[1:DP,:] = Z_bar_true

## Generate x regressors
# Use word counts to define x so that it is correlated with topics
x1 = zeros(N,1)
x2 = randn(N,1)
x1[1:D] = group_mean(hcat(Array(Float64.(word_counts[:,1]))), docidx_dtm[1:DP])
x1 = (x1.-mean(x1))./std(x1)
x = Array{Float64,2}(hcat(x1,x2))

## Aggregate interactions to paragraph level
Z_bar_all =reshape(vec(group_mean(Z_bar_all, docidx_dtm)),(N,K))

regressors = hcat(Z_bar_all, x)

## Generate outcome variable
p1 = exp.(regressors*ω_true) ./(1 .+ exp.(regressors*ω_true))
y = Array{Int64}([rand()<p1[ii] for ii in 1:N])
y_lin = Z_bar_all*ω_z_true + x*ω_x_true


"""
Test whether synthetic data gives correct coefficients in Logistic regression with true data
"""
## With true data and correct model
regressors = hcat(Z_bar_all, x)
logit_true_all = fit(GeneralizedLinearModel, regressors, y, Binomial(), LogitLink())
predictions = GLM.predict(logit_true_all, regressors)
prediction_class = [if x < 0.5 0 else 1 end for x in predictions]
correct_true_all = mean(y .== prediction_class)
mse_true_all = mean((predictions.- y).^2)

## Just text
regressors = hcat(Z_bar_all)
logit_true_nox = fit(GeneralizedLinearModel, regressors, y, Binomial(), LogitLink())
predictions = GLM.predict(logit_true_nox, regressors)
prediction_class = [if x < 0.5 0 else 1 end for x in predictions]
correct_true_nox = mean(y .== prediction_class)
mse_true_nox = mean((predictions.- y).^2)

### Just x
regressors = hcat(ones(N),x)
logit_true_noz = fit(GeneralizedLinearModel, regressors, y, Binomial(), LogitLink())
predictions = GLM.predict(logit_true_noz, regressors)
prediction_class = [if x < 0.5 0 else 1 end for x in predictions]
correct_true_noz = mean(y .== prediction_class)
mse_true_noz = mean((predictions.- y).^2)


"""
Convert the documents to a Document-Term-Matrix
"""
## Use TextAnalysis package to create a DTM
docs = StringDocument.(docs_all)
crps = Corpus(docs)
update_lexicon!(crps)
dtm_sparse = DocumentTermMatrix(crps)

## Save vocabulary for figures
vocab = dtm_sparse.terms


"""
Generate word count variable from DTM
"""
## Define wordlist
list1 = ["1","test"]

## Get wordcounts
list1_counts = hcat(Float64.(wordlistcounts(dtm_sparse.dtm,vocab,list1)))
list1_counts = group_mean(list1_counts, docidx_dtm)
list1_score = (list1_counts.-mean(list1_counts))./std(list1_counts)
#list1_score = group_mean(list1_score,doc_idx)
@assert list1_score == x1


"""
Save the synthetic data to csv files
"""
## DataFrame for regression data
df = DataFrame(doc_id = docidx_vars,
               y = y,
               x1 = x[:,1],
               x2 = x[:,2],
               Z_bar1 = Z_bar_all[:,1],
               Z_bar2 = Z_bar_all[:,2],
               Z_bar3 = Z_bar_all[:,3])
if save_files
    dtmtodfs(dtm_sparse.dtm, docidx_dtm, vocab, save_dir = "data/class_")
    CSV.write("data/synth_class_data.csv", df)
end




"""
Split into training and test sets (by doc_idx) and convert to BTRRawData structure
"""
## Extract sparse matrix and create BTRRawData structure(s)
dtm_in = dtm_sparse.dtm
train_data, test_data = btc_traintestsplit(dtm_in, docidx_dtm, docidx_vars, y, vocab, x = x,
    train_split = 0.75, shuffle_obs = true)
# Alternatively, can convert the entier set to BTRRawData with
all_data = DocStructs.BTCRawData(dtm_in, docidx_dtm, docidx_vars, y, x, vocab)
## Visualise the training-test split
histogram(train_data.docidx_dtm, bins = 1:N, label = "training set",
    xlab = "Observation", ylab= "Paragraphs", c=1, lc=nothing)
display(histogram!(test_data.docidx_dtm, bins = 1:N, label = "test set", c=2, lc=nothing))
if save_files; savefig("figures/synth_BTC/synth_class_trainsplit.pdf"); end;



"""
Set priors and estimation optioncs here to be consistent across models
"""
## Initialiase estimation options
btcopts = BTCOptions()

## Number of topics
btcopts.ntopics = 3

## LDA priors
btcopts.α=1.
btcopts.η=1.

## Number of iterations cross-validation
btcopts.E_iters = 50 # E-step iterations (sampling topic assignments, z)
btcopts.M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
btcopts.EM_iters = 10 # Maximum possible EM iterations (will stop here if no convergence)
btcopts.burnin = 10 # Burnin for Gibbs samplers
btcopts.CVEM = :none # Split for separate E and M step batches (if batch = true)
btcopts.CVEM_split = 0.5 # Split for separate E and M step batches (if batch = true)

## Convergence
btcopts.crossent_conv = 1 # Number of previous periods to average over for crossent convergence
btcopts.ω_tol = 0.005 # Convergence tolerance for regression coefficients ω
btcopts.rel_tol = true # Whether to use a relative convergence criteria rather than just absolute







"""
Run some text-free regressions for benchmarking
"""
## Define regressors
regressors_train = hcat(ones(size(train_data.x,1)),train_data.x)
regressors_test = hcat(ones(size(test_data.x,1)),test_data.x)

## Logitistic Regression
logit_notext = fit(GeneralizedLinearModel, regressors_train, train_data.y, Binomial(), LogitLink())
display(logit_notext)
# Assess out-of-sample performance
prediction_notext = GLM.predict(logit_notext,regressors_test)
prediction_class_notext = [if x < 0.5 0 else 1 end for x in prediction_notext];
correct_notext = mean(test_data.y .== prediction_class_notext)
mse_notext = mean((prediction_notext.- test_data.y).^2)



"""
Estimate BTC
"""
## Include x regressors by changing the options
btcopts.xregs = [1,2]
btcopts.interactions = Array{Int64}([])

## Initialise BTRModel object
btccrps_tr = create_btccrps(train_data, btcopts.ntopics)
btccrps_ts = create_btccrps(test_data, btcopts.ntopics)
btcmodel = BTCModel(crps = btccrps_tr, options = btcopts)

## Estimate BTR with EM-Gibbs algorithm
btcopts.CVEM = :obs
btcopts.CVEM_split = 0.5
btcmodel = BTCemGibbs(btcmodel)

## Plot results
#logit_reg = fit(GeneralizedLinearModel, btcmodel.regressors, btcmodel.y, Binomial(), LogitLink())
topic_order = synth_reorder_topics(btcmodel.β)
plt = synth_data_plot(btcmodel.β, btcmodel.ω, btcmodel.Σ, true_ω = ω_true,
    topic_ord = topic_order, plt_title = "", legend = false,
    left_mar = 3,top_mar = 3, bottom_mar = 0, ticksize = 12, labelsize = 25)
if save_files; plot!(size = (300,300)); savefig("figures/synth_BTC/synth_class_BTC.pdf"); end;

## Out of sample prediction in test set
btc_predicts = BTCpredict(btccrps_ts, btcmodel)
# Performance
correct_btc = mean(test_data.y .== btc_predicts.y_pred)
mse_btc = mean((test_data.y .- btc_predicts.p_pred).^2)
btc_predicts.pplxy




"""
Estimate 2 stage LDA then Logistic Regression (BLR)
    In the synthetic data this does about as well as BTC because the
    text is generated from an LDA model.
"""
# Use the same options as the BTC, but might want to tweak the number of iterations as there's only one step
ldaopts = deepcopy(btcopts)
ldaopts.fullGibbs_iters = 100
ldaopts.fullGibbs_thinning = 2
ldaopts.burnin = 50

ldamodel = BTCModel(crps = btccrps_tr, options = ldaopts)
## Estimate LDA model on full training set
ldamodel  = LDAGibbs(ldamodel)

## Logistic regression on training set
logit_lda = fit(GeneralizedLinearModel, ldamodel.regressors, ldamodel.y, Binomial(), LogitLink())
ldamodel.ω = coef(logit_lda)
ldamodel.Σ = I(length(ldamodel.ω)).*stderror(logit_lda)

## Plot results
topic_order = synth_reorder_topics(ldamodel.β)
# Without interactions
plt = synth_data_plot(ldamodel.β, ldamodel.ω, ldamodel.Σ, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(xticklabel = false)
if save_files; plot!(size = (300,300)); savefig("figures/synth_BTC/synth_class_LDA_Log.pdf"); end;


## Out of sample prediction
lda_predicts = BTCpredict(btccrps_ts, ldamodel)
# Performance
correct_lda = mean(test_data.y .== lda_predicts.y_pred)
mse_lda = mean((test_data.y .- lda_predicts.p_pred).^2)
lda_predicts.pplxy


"""
Not sure how you'd do the 2-stage way for classification, so just do standard sLDA
"""
## Standard sLDA
sldaopts = deepcopy(btcopts)
sldaopts.xregs = []
sldaopts.interactions = []

## Initialise BTRModel object
sldamodel = BTCModel(crps = btccrps_tr, options = sldaopts)

## Estimate sLDA on residuals
sldamodel = BTCemGibbs(sldamodel)

## Plot results
topic_order = synth_reorder_topics(sldamodel.β)
plt = synth_data_plot(sldamodel.β, sldamodel.ω, sldamodel.Σ, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)

if save_files; plot!(size = (300,300)); savefig("figures/synth_LR_sLDA.pdf"); end;

## Out of sample prediction
slda_predicts = BTCpredict(btccrps_ts, sldamodel)
# Performance
correct_nox = mean(test_data.y .== slda_predicts.y_pred)
mse_nox = mean((test_data.y .- slda_predicts.p_pred).^2)
slda_predicts.pplxy


"""
Compare out of sample performance for each model
"""
## Restrict range so that figure isn't too busy
plot_range = sample(1:test_data.N, 100)
# True data
plot(test_data.y[plot_range], linestyle = :solid,
    label = join(["True data"]), title = "")
# BTC with interactions
plot!(btc_predicts.p_pred[plot_range], linestyle = :solid,
    label = join(["BTC (", round(mse_btc,digits = 3), ")"]))
# Logisitic sLDA with no x
plot!(slda_predicts.p_pred[plot_range], linestyle = :dashdot,
    label = join(["Log sLDA (", round(mse_nox,digits = 3), ")"]))
# Logistic with x only
plot!(prediction_notext[plot_range], linestyle = :dot,
    label = join(["Logisitic (", round(mse_notext,digits = 3), ")"]))
if save_files; savefig("figures/synth_class_mse_comparison.pdf"); end;


subdirectory = "data/multipleruns/BTC/run_"
nruns = 10
opts = btcopts
BTC_multipleruns(train_data, test_data, btcopts, nruns, subdirectory)
