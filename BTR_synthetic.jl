"""
Implementation of BTR with Gibbs sampling for a toy dataset
"""
## Set working directory
cd("/Users/julianashwin/Documents/GitHub/BTR.jl/")


"""
Load functions and necessary packages
    To make sure the latest version of the package is used run
    pkg> dev /Users/julianashwin/Documents/GitHub/BTR.jl
    or
    pkg> dev https://github.com/julianashwin/BTR.jl

"""

using Revise, BTR
#include("src/BTR_dev.jl")
using TextAnalysis, DataFrames, CSV, Plots
using StatsPlots, StatsBase, Plots.PlotMeasures, Distributions, Random


"""
Plotting options
"""
# This sets the plotting backend: gr() is faster, pyplot() is prettier
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
D = 5000 # number of documents
Pd = 4 # number of paragraphs
Np = 25 # number of words per document
N = D+10 # total observations (N>=D)
NP = N*Pd # total number of paragraphs
DP = D*Pd # total number of non-empty paragraphs

## Parameters of true regression model
ω_z_true = [-1.,0.,1.] # coefficients on topics
#ω_zx_true = [2.,0.,1.] # coefficients on (topic,x) interactions
ω_x_true = [2., 0.] # coefficients on (topic,x) interactions
#ω_true = cat(ω_z_true, ω_zx_true, ω_x_true, dims = 1) # all coefficients
ω_true = cat(ω_z_true, ω_x_true, dims = 1) # all coefficients
σ_y_true = 0.05 # residual variance

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
if save_files; plot!(size =(200,300)); savefig("figures/synth_true_beta.pdf"); end;

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
#x1 = zeros(NP,1)
x1 = zeros(N,1)
x2 = randn(N,1)
#x2 = reshape(repeat(x2, inner = Pd),(NP,1))
x1[1:D] = group_mean(hcat(Array(Float64.(word_counts[:,1]))), docidx_dtm[1:DP])
x1 = (x1.-mean(x1))./std(x1)
#x1 = reshape(repeat(vec(group_mean(x1, doc_idx)), inner = Pd),(NP,1))
x = Array{Float64,2}(hcat(x1,x2))
ϵ_true = rand(Normal(0,sqrt(σ_y_true)), N)

## Generate interaction regressors
#inter_effects = zeros(NP,(K*size(x1,2)))
#for jj in 1:size(x1,2)
#    col_range= (jj*K-K+1):(jj*K)
#    inter_effects[:,col_range] = Z_bar_all.*vec(x[:,jj])
#end

## Aggregate interactions to paragraph level
# Aggregate inter_effects and Z_bar (straight mean is fine as long as all docs have same number of paras)
#inter_effects = reshape(vec(group_mean(inter_effects, doc_idx)),(N,K))
Z_bar_all =reshape(vec(group_mean(Z_bar_all, docidx_dtm)),(N,K))


## Generate outcome variable
#y = Z_bar_all*ω_z_true + inter_effects*ω_zx_true + x2*ω_x_true + ϵ_true
y = Z_bar_all*ω_z_true + x*ω_x_true + ϵ_true
# ϵ_true, y, Z_bar_all,inter_effects and the xs that will not be interacted should be the same within groups

"""
Test whether synthetic data gives correct coefficients in OLS regression with true data
"""
## Just observations with documents
#regressors = hcat(Z_bar_all[1:D,:], inter_effects[1:D,:],x2[1:D,:])
regressors = hcat(Z_bar_all[1:D,:], x[1:D,:])
ols_coeffs = inv(transpose(regressors)*regressors)*(transpose(regressors)*y[1:D])
#display(ols_coeffs)
mse_true = mean((y[1:D] .- regressors*ols_coeffs).^2)

regressors = hcat(Z_bar_all[1:D,:])
ols_coeffs = inv(transpose(regressors)*regressors)*(transpose(regressors)*y[1:D])
#display(ols_coeffs)
mse_nox = mean((y[1:D] .- regressors*ols_coeffs).^2)

regressors = hcat(ones(D),x[1:D,:])
ols_coeffs = inv(transpose(regressors)*regressors)*(transpose(regressors)*y[1:D])
#display(ols_coeffs)
mse_noz = mean((y[1:D] .- regressors*ols_coeffs).^2)


## All observations
#regressors = hcat(Z_bar_all, inter_effects,x2)
regressors = hcat(Z_bar_all, x)
ols_coeffs = inv(transpose(regressors)*regressors)*(transpose(regressors)*y)
#display(ols_coeffs)
mse_insample = mean((y .- regressors*ols_coeffs).^2)

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
Split into training and test sets (by doc_idx) and convert to BTRRawData structure
"""
## Extract sparse matrix and create BTRRawData structure(s)
dtm_in = dtm_sparse.dtm
train_data, test_data = btr_traintestsplit(dtm_in, docidx_dtm, docidx_vars, y, vocab, x = x,
    train_split = 0.75, shuffle_obs = true)
# Alternatively, can convert the entier set to BTRRawData with
all_data = DocStructs.BTRRawData(dtm_in, docidx_dtm, docidx_vars, y, x, vocab)
## Visualise the training-test split
histogram(train_data.docidx_dtm, bins = 1:N, label = "training set",
    xlab = "Observation", ylab= "Paragraphs", c=1, lc=nothing)
display(histogram!(test_data.docidx_dtm, bins = 1:N, label = "test set", c=2, lc=nothing))
if save_files; savefig("figures/synth_trainsplit.pdf"); end;


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
    dtmtodfs(dtm_sparse.dtm, docidx_dtm, vocab, save_dir = "data")
    CSV.write("data/synth_data.csv", df)
end


"""
Set priors and estimation optioncs here to be consistent across models
"""
## Initialiase estimation options
btropts = BTROptions()
## Number of topics
btropts.ntopics = 3
## LDA priors
btropts.α=1.
btropts.η=1.
## BLR priors
btropts.μ_ω = 0. # coefficient mean
btropts.σ_ω = 2. # coefficient variance
btropts.a_0 = 4. # residual shape: higher moves mean closer to zero
btropts.b_0 = 0.2 # residual scale: higher is more spread out
# Plot the prior distribution for residual variance (in case unfamiliar with InverseGamma distributions)
# mean will be b_0/(a_0 - 1)
plot(InverseGamma(btropts.a_0, btropts.b_0), xlim = (0,1), title = "Residual variance prior",
    label = "Prior on residual variance")
display(scatter!([σ_y_true],[0.],label = "True residual variance"))
if save_files; savefig("figures/synthetic_IGprior.pdf"); end;

## Number of iterations and convergence tolerance
btropts.E_iters = 100 # E-step iterations (sampling topic assignments, z)
btropts.M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
btropts.EM_iters = 10 # Maximum possible EM iterations (will stop here if no convergence)
btropts.CVEM = :none # Split for separate E and M step batches (if batch = true)
btropts.CVEM_split = 0.5 # Split for separate E and M step batches (if batch = true)
btropts.burnin = 10 # Burnin for Gibbs samplers
btropts.ω_tol = 0.005 # Convergence tolerance for regression coefficients ω
btropts.rel_tol = true # Whether to use a relative convergence criteria rather than just absolute




"""
Convert to BTRCorpus objects
"""
btrcrps_tr = create_btrcrps(train_data, btropts.ntopics)
btrcrps_ts = create_btrcrps(test_data, btropts.ntopics)
# Can also create_btrdocs without the BTRRawData structure with
#btrdocs_tr, topics_tr, doclabels_tr = create_btrdocs(dtm_in, doc_idx, y, x, ntopics)



"""
Run some text-free regressions for benchmarking
"""
## Define regressors
regressors_train = hcat(ones(size(train_data.x,1)),train_data.x)
regressors_test = hcat(ones(size(test_data.x,1)),test_data.x)

## OLS
ols_coeffs = inv(transpose(regressors_train)*regressors_train)*(transpose(regressors_train)*train_data.y)
mse_ols = mean((test_data.y .- regressors_test*ols_coeffs).^2)

## Bayesian linear regression
blr_coeffs_post, σ2_post = BLR_Gibbs(train_data.y, regressors_train, iteration = btropts.M_iters,
    m_0 = btropts.μ_ω, σ_ω = btropts.σ_ω, a_0 = btropts.a_0, b_0 = btropts.b_0)
blr_coeffs = Array{Float64,1}(vec(mean(blr_coeffs_post, dims = 2)))

predict_blr = regressors_test*blr_coeffs
mse_blr = mean((test_data.y .- predict_blr).^2)


"""
Estimate BTR
"""
## Include x regressors by changing the options
btropts.xregs = [1,2]
btropts.interactions = [2]

## Initialise BTRModel object
btrmodel = BTRModel(crps = btrcrps_tr, options = btropts)

## Estimate BTR with EM-Gibbs algorithm
btropts.CVEM = :none
btropts.CVEM_split = 0.5
btrmodel = BTRemGibbs(btrmodel)

## Plot results
topic_order = synth_reorder_topics(btrmodel.β)
plt = synth_data_plot(btrmodel.β, btrmodel.ω_post, true_ω = ω_true,
    topic_ord = topic_order, plt_title = "", legend = false,
    left_mar = 3,top_mar = 3, bottom_mar = 0, ticksize = 12, labelsize = 25)
if save_files; savefig("figures/synth_BTR.pdf"); end;

## Out of sample prediction in test set
btr_predicts = BTRpredict(btrcrps_ts, btrmodel)
#btr_predicts.crps.topics = gettopics(btrcrps_ts.docs)
mse_btr = mean((btr_predicts.y .- btr_predicts.y_pred).^2)



"""
Estimate 2 stage LDA then Bayesian Linear Regression (BLR)
    In the synthetic data this does about as well as BTR because the
    text is generated from an LDA model.
"""
# Use the same options as the BTR, but might want to tweak the number of iterations as there's only one step
ldaopts = deepcopy(btropts)
ldaopts.fullGibbs_iters = 1000
ldaopts.fullGibbs_thinning = 2
ldaopts.burnin = 50

ldamodel = BTRModel(crps = btrcrps_tr, options = ldaopts)
## Estimate LDA model on full training set
ldamodel  = LDAGibbs(ldamodel)

## Bayesian linear regression on training set
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(ldamodel.y, ldamodel.regressors,
    m_0 = ldaopts.μ_ω, σ_ω = ldaopts.σ_ω, a_0 = ldaopts.a_0, b_0 = ldaopts.b_0)
ldamodel.ω = blr_ω
ldamodel.ω_post = blr_ω_post

## Plot results
topic_order = synth_reorder_topics(ldamodel.β)
# Without interactions
plt = synth_data_plot(ldamodel.β, ldamodel.ω_post, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(size = (300,300))
plot!(xticklabel = false)
if save_files; savefig("figures/synth_LDA_LR.pdf"); end;


## Out of sample prediction
lda_predicts = BTRpredict(btrcrps_ts, ldamodel)

## Calculate MSE
mse_lda_blr = mean((lda_predicts.y .- lda_predicts.y_pred).^2)



"""
Estimate 2 stage BLR then supervised LDA (sLDA) on residuals
"""
## Residualise first
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(train_data.y, hcat(ones(size(train_data.x,1)),train_data.x),
    m_0 = ldaopts.μ_ω, σ_ω = ldaopts.σ_ω, a_0 = ldaopts.a_0, b_0 = ldaopts.b_0)
# Get the y residualised on x
resids_blr_train = train_data.y - hcat(ones(size(train_data.x,1)),train_data.x)*blr_ω
# Create corpus with residual as y
btrcrps_tr_slda = deepcopy(btrcrps_tr)
for ii in 1:btrcrps_tr_slda.N
    btrcrps_tr_slda.docs[ii].y = resids_blr_train[ii]
end

## Set options sLDA on residuals
slda1opts = deepcopy(btropts)
slda1opts.xregs = []
slda1opts.interactions = []

## Initialise BTRModel object
slda1model = BTRModel(crps = btrcrps_tr_slda, options = slda1opts)

## Estimate sLDA on residuals
slda1model = BTRemGibbs(slda1model)

## Plot results
topic_order = synth_reorder_topics(slda1model.β)
plt = synth_data_plot(slda1model.β, slda1model.ω_post, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(size = (300,300))
if save_files; savefig("figures/synth_LR_sLDA.pdf"); end;

## Out of sample prediction
y_stage1_test = hcat(ones(size(test_data.x,1)),test_data.x)*blr_ω
resids_test = test_data.y - y_stage1_test
# Create corpus with residual as y
btrcrps_ts_slda = deepcopy(btrcrps_ts)
for ii in 1:btrcrps_ts_slda.N
    btrcrps_ts_slda.docs[ii].y = resids_test[ii]
end

slda1_predicts = BTRpredict(btrcrps_ts_slda, slda1model)

mse_blr_slda = mean((slda1_predicts.y .-slda1_predicts.y_pred).^2)


"""
Estimate 2 stage slDA then BLR on residuals
"""
## Set options sLDA on residuals
slda2opts = deepcopy(btropts)
slda2opts.xregs = []
slda2opts.interactions = []

## Initialise BTRModel object
slda2model = BTRModel(crps = btrcrps_tr, options = slda2opts)

## Estimate sLDA on residuals
slda2model = BTRemGibbs(slda2model)

## Plot results
topic_order = synth_reorder_topics(slda2model.β)
# Without interactions
plt = synth_data_plot(slda2model.β, slda2model.ω_post, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(size = (300,300))
if save_files; savefig("figures/synth_sLDA_LR.pdf"); end;

## Identify residuals to train second stage regression
residuals_slda = train_data.y .- slda2model.regressors*slda2model.ω

## Bayesian linear regression on training set
# Create regressors
regressors_slda = hcat(ones(size(train_data.x,1)),train_data.x)
# No fixed effect, no batch
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(residuals_slda, regressors_slda,
    m_0 = slda2opts.μ_ω, σ_ω = slda2opts.σ_ω, a_0 = slda2opts.a_0, b_0 = slda2opts.b_0)

## Out of sample
slda2_predicts = BTRpredict(btrcrps_ts, slda2model)

# Second stage regressors
regressors_test = hcat(ones(size(test_data.x,1)),test_data.x)
# Add y prediction from slda to portion of residual explained by BLR
y_pred_slda_blr = slda2_predicts.y_pred + regressors_test*blr_ω
# Compute MSE
mse_slda_blr = mean((test_data.y .- y_pred_slda_blr).^2)




"""
Compare out of sample performance for each model
"""
## Restrict range so that figure isn't too busy
plot_range = 500:550
# True data
plot(test_data.y[plot_range], linestyle = :solid,
    label = join(["True data"]), title = "")
# BTR with interactions
plot!(btr_predicts.y_pred[plot_range], linestyle = :solid,
    label = join(["BTR (", round(mse_btr,digits = 3), ")"]))
# BLR then SLDA
plot!((slda1_predicts.y_pred .+ y_stage1_test)[plot_range], linestyle = :dashdot,
    label = join(["LR + sLDA (", round(mse_blr_slda,digits = 3), ")"]))
# BLR then SLDA
plot!(y_pred_slda_blr[plot_range], linestyle = :dot,
    label = join(["sLDA + LR (", round(mse_slda_blr,digits = 3), ")"]))
# BlR without documents
plot!(predict_blr[plot_range], linestyle = :dash,
    label = join(["BLR (", round(mse_blr,digits = 3), ")"]))
if save_files; savefig("figures/synth_mse_comparison.pdf"); end;


subdirectory = "data/multipleruns/BTR/run_"
nruns = 10
BTR_multipleruns(train_data, test_data, btropts, nruns, subdirectory)
