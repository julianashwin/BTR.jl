"""
Implementation of BTR with Gibbs sampling for a toy dataset
"""
## Set working directory (needs to have the src folder in it)
cd("/Users/julianashwin/Documents/GitHub/BTR.jl/")


"""
Load functions and necessary packages
    You'll need to install the following packages if you don't have them:
        SparseArrays, LinearAlgebra, Random, Distributions, DataFrames, GLM
        TextAnalysis, Plots, StatsPlots, ProgressMeter, CSV, KernelDensity,
        StatsBase, ColorSchemes, Plotly
    You can do that with Pkg.add("name")
"""

"""
To make sure the latest version of the package is used run
pkg> dev /Users/julianashwin/Documents/GitHub/BTR.jl
or
pkg> dev https://github.com/julianashwin/BTR.jl

"""

#using BTR
include("src/BTR_dev.jl")
using TextAnalysis, DataFrames, CSV
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
D = 100 # number of documents
Pd = 4 # number of paragraphs
Np = 10 # number of words per document
N = D # total observations (N>=D)
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
display(plot!(size =(200,300)))
if save_files; savefig("figures/synth_true_beta.pdf"); end;

## Generate string documents and topic assignments
docs, Z_true, topic_counts, word_counts = generate_docs(DP, Np, K, θ_true, β_true)
docs_all = repeat([""], NP)
docs_all[1:DP] = docs

## Generate document indicator for each document
doc_idx = repeat(1:N,inner=Pd)

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
x1[1:D] = group_mean(hcat(Array(Float64.(word_counts[:,1]))), doc_idx)
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
# Identify size and first member of each group
Nps = counts(doc_idx)
first_paras = findfirst_group(doc_idx)
# Aggregate inter_effects and Z_bar (straight mean is fine as long as all docs have same number of paras)
#inter_effects = reshape(vec(group_mean(inter_effects, doc_idx)),(N,K))
Z_bar_all =reshape(vec(group_mean(Z_bar_all, doc_idx)),(N,K))


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
display(ols_coeffs)
mse_true = mean((y .- regressors*ols_coeffs).^2)

regressors = hcat(Z_bar_all[1:D,:])
ols_coeffs = inv(transpose(regressors)*regressors)*(transpose(regressors)*y[1:D])
display(ols_coeffs)
mse_nox = mean((y .- regressors*ols_coeffs).^2)

regressors = hcat(ones(D),x)
ols_coeffs = inv(transpose(regressors)*regressors)*(transpose(regressors)*y[1:D])
display(ols_coeffs)
mse_noz = mean((y .- regressors*ols_coeffs).^2)


## All observations
#regressors = hcat(Z_bar_all, inter_effects,x2)
regressors = hcat(Z_bar_all, x)
ols_coeffs = inv(transpose(regressors)*regressors)*(transpose(regressors)*y)
display(ols_coeffs)
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
list1 = ["1","test"]

list1_counts = hcat(Float64.(wordlistcounts(dtm_sparse.dtm,vocab,list1)))
list1_counts = group_mean(list1_counts, doc_idx)
list1_score = (list1_counts.-mean(list1_counts))./std(list1_counts)
#list1_score = group_mean(list1_score,doc_idx)
@assert list1_score == x1

"""
Save the synthetic data to csv files
"""
## DataFrame for regression data
df = DataFrame(#doc_id = doc_idx,
               y = y,
               x1 = x[:,1],
               x2 = x[:,2],
               #theta1 = θ_all[1,:],
               #theta2 = θ_all[2,:],
               #theta3 = θ_all[3,:],
               Z_bar1 = Z_bar_all[:,1],
               Z_bar2 = Z_bar_all[:,2],
               Z_bar3 = Z_bar_all[:,3],
               #text = docs_all
               )
if save_files
    dtmtodfs(dtm_sparse.dtm, doc_idx, vocab, save_dir = "data")
    CSV.write("data/synth_data.csv", df)
end


"""
Split into training and test sets (by doc_idx) and convert to BTRRawData structure
"""
dtm_in = dtm_sparse.dtm
train_data, test_data = btr_traintestsplit(dtm_in, doc_idx, y, x = x,
    train_split = 0.75, shuffle_obs = true)
# Alternatively, can convert the entier set to BTRRawData with
# train_data = DocStructs.BTRRawData(dtm_in, doc_idx, y, x)
## Visualise the training-test split
histogram(train_data.doc_idx, bins = 1:N, label = "training set",
    xlab = "Observation", ylab= "Paragraphs", c=1, lc=1)
display(histogram!(test_data.doc_idx, bins = 1:N, label = "test set", c=2, lc=2))
if save_files; savefig("figures/synth_trainsplit.pdf"); end;

"""
Set priors and hyperparameters here to be consistent across models
"""
## Number of topics
ntopics = 3

## LDA priors
α=1.
η=1.

## BLR priors
μ_ω = 0. # coefficient mean
σ_ω = 1. # coefficient variance
a_0 = 3. # residual shape: higher moves mean closer to zero
b_0 = 0.2 # residual scale: higher is more spread out
# Plot the prior distribution for residual variance (in case unfamiliar with InverseGamma distributions)
# mean will be b_0/(a_0 - 1)
plot(InverseGamma(a_0, b_0), xlim = (0,1), title = "Residual variance prior",
    label = "Prior on residual variance")
display(scatter!([σ_y_true],[0.],label = "True residual variance"))
if save_files; savefig("figures/synthetic_IGprior.pdf"); end;



"""
Convert to arrays of BTRParagraphDocument
"""
btrdocs_tr, topics_tr, doclabels_tr = create_btrdocs(train_data, ntopics)
btrdocs_ts, topics_ts, doclabels_ts = create_btrdocs(test_data, ntopics)
# Can also create_btrdocs without the BTRRawData structure with
#btrdocs_tr, topics_tr, doclabels_tr = create_btrdocs(dtm_in, doc_idx, y, x, ntopics)


"""
Set the estimation options here, to be consistent across models
"""
## Number of iterations and convergence tolerance
E_iters = 500 # E-step iterations (sampling topic assignments, z)
M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
EM_iters = 100 # Maximum possible EM iterations (will stop here if no convergence)
EM_split = 0.5 # Split for separate E and M step batches (if batch = true)
burnin = 100 # Burnin for Gibbs samplers
ω_tol = 0.005 # Convergence tolerance for regression coefficients ω


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
blr_coeffs_post, σ2_post = BLR_Gibbs(train_data.y, regressors_train, iteration = M_iters,
    m_0 = μ_ω, V_0 = σ_ω, a_0 = a_0, b_0 = b_0)
blr_coeffs = Array{Float64,1}(vec(mean(blr_coeffs_post, dims = 2)))

predict_blr = regressors_test*blr_coeffs
mse_blr = mean((test_data.y .- predict_blr).^2)

topics_tr1 = gettopics(btrdocs_tr,ntopics)
topics_tr1 = gettopics(btrdocs_tr,4)


"""
Estimate BTR
"""
btrmodel = DocStructs.BTRModel(btrdocs_tr, topics_tr, vocab, K, zeros(K,V), zeros(K,length(btrdocs_tr)), zeros(K,100), zeros(100))
btrmodel = DocStructs.BTRModel(btrdocs_tr, length(vocab), ntopics)

dtm_in = SparseMatrixCSC{Int64,Int64}(dtm_in)
ntopics = Int64(ntopics)
y = Array{Float64,1}(train_data.y);
x = Array{Float64,2}(train_data.x) # = zeros(1,1),
doc_idx = Array{Int64,1}(train_data.doc_idx) # = [0],
σ2 = Float64(0.) # = 0.,
α  = Float64(α) # =1.,
η = Float64(η) # = 1.,
σ_ω = Float64(σ_ω) # = 1.,
μ_ω = Float64(μ_ω) # = 0.,
a_0 = Float64(a_0) # = 0.,
b_0 = Float64(b_0)  # = 0.,
E_iters = Int64(E_iters) # = 100,
M_iters = Int64(M_iters) # = 500,
EM_iters = Int64(EM_iters) # = 10,
burnin = Int64(burnin) # = 10,
topics_init = Array(Array{Main.DocStructs.Topic,1}(undef, 1)) # = Array{Main.DocStructs.Topic,1}(undef, 1),
docs_init = Array(Array{Main.DocStructs.TopicBasedDocument,1}(undef, 1)) # = Array{Main.DocStructs.TopicBasedDocument,1}(undef, 1),
ω_init = Array{Float64,1}(zeros(1)) # = zeros(1),
ω_tol = Float64(0.01) # = 0.01,
rel_tol = Bool(true) # = false,
interactions = Array{Int64,1}([]) # =Array{Int64,1}([]),
xreg = Array{Int64,1}([1,2]) # Array{Int64,1}([])
CVEM = Bool(false) # =false,
CVEM_split = Float64(0.75) # = 0.75,
leave_one_topic_out = Bool(false) # =false,
plot_ω = Bool(true) # = false

btrdocs = btrdocs_tr
topics = gettopics(btrdocs, ntopics)


@assert length(interactions) <= length(xreg) "Can't have more interactions than x variables"
J = length(xreg)
J_inter = length(interactions)*ntopics
J_nointer = J - length(interactions) # Can we drop this?
no_interactions = setdiff(1:J, interactions)

N = length(btrmodel.docs)
V = length(btrmodel.vocab)










"""
Estimate BTR
"""
## Estimate BTR
@time results_btr =
    BTR_EMGibbs(dtm_train, ntopics, y_train, x = x_train,
        α = α, η = η, σ_ω = σ_ω, a_0 = a_0, b_0 = b_0,
        batch = true, ω_tol = ω_tol, rel_tol=true, plot_ω=true,
        E_iteration = E_iteration, EM_iteration = EM_iteration,
        M_iteration = M_iteration, EM_split = EM_split, burnin = burnin)
topic_order = synth_reorder_topics(results_btr.β)
plt = synth_data_plot(results_btr.β, results_btr.ω_post, true_ω = ω_true,
    topic_ord = topic_order, plt_title = "", legend = false,
    left_mar = 3,top_mar = 3, bottom_mar = 0, ticksize = 12, labelsize = 25)


@time results_btr =
    BTR_EMGibbs_paras(dtm_train, ntopics, y_train, x = x_train,
        α = α, η = η, σ_ω = σ_ω, a_0 = a_0, b_0 = b_0,
        doc_idx = doc_idx_train,
        batch = true, ω_tol = ω_tol, rel_tol=true, plot_ω=true,
        E_iteration = E_iteration, EM_iteration = EM_iteration,
        M_iteration = M_iteration, EM_split = EM_split, burnin = burnin)
topic_order = synth_reorder_topics(results_btr.β)
plt = synth_data_plot(results_btr.β, results_btr.ω_post, true_ω = ω_true,
    topic_ord = topic_order, plt_title = "", legend = false,
    left_mar = 3,top_mar = 3, bottom_mar = 0, ticksize = 12, labelsize = 25)



ω_kk = sort(results_btr.ω_post[4,:])
lo = ω_kk[Int(round(0.025*M_iteration))]
hi = ω_kk[Int(round(0.975*M_iteration))]
mid = mean(ω_kk)



## Plot and save estimated model
topic_order = synth_reorder_topics(results_btr.β)
plt = synth_data_plot(results_btr.β, results_btr.ω_post, true_ω = ω_true,
    topic_ord = topic_order, plt_title = "", legend = false,
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(size = (300,300))
plot!(xticklabel = false)
savefig("figures/synth_BTR.pdf")

## Out of sample prediction in test set
predict_btr =
    BTR_Gibbs_predict(dtm_test, ntopics, results_btr.β, results_btr.ω,
    x = x_test, α = α, Σ = results_btr.Σ,
    E_iteration = M_iteration, burnin =burnin)
mse_btr = mean((y_test .- predict_btr.y_pred).^2)




"""
Estimate 2 stage LDA then Bayesian Linear Regression (BLR)
    In the synthetic data this does as well or better than BTR as the
    data is generated from an LDA model.
"""
## Estimate LDA model on full training set
@time results_lda  = LDA_Gibbs(dtm_train, ntopics, α = α, η = η,
    iteration = E_iteration, burnin=burnin)


## Create regressors with an without fixed effects and interaction effects
inter_effects_lda_train = create_inter_effects(results_lda.Z_bar,x_train,ntopics)
regressors_train = hcat(transpose(results_lda.Z_bar), x_train)
regressors_inter_train = hcat(transpose(results_lda.Z_bar), inter_effects_lda_train)

## Bayesian linear regression on training set
# No interactions
blr_coeffs_post, σ2_post = BLR_Gibbs(y_train, regressors_train, iteration = M_iteration,
    m_0 = μ_ω, V_0 = σ_ω, a_0 = a_0, b_0 = b_0)
blr_coeffs = Array{Float64,1}(vec(mean(blr_coeffs_post, dims = 2)))

## Plot results
topic_order = synth_reorder_topics(results_lda.β)
# Without interactions
plt = synth_data_plot(results_lda.β, blr_coeffs_post, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(size = (300,300))
plot!(xticklabel = false)
savefig("figures/synth_LDA_LR.pdf")


## Out of sample prediction
Z_bar_lda_test =
    LDA_Gibbs_predict(dtm_test, ntopics, results_lda.β,
    α = α, iteration = E_iteration, burnin = burnin)

## Create regressors with an without fixed effects and interaction effects
inter_effects_lda_test = create_inter_effects(Z_bar_lda_test,x_test,ntopics)
regressors_test = hcat(transpose(Z_bar_lda_test), x_test)
regressors_inter_test = hcat(transpose(Z_bar_lda_test), inter_effects_lda_test)

## Calculate MSE
mse_lda_blr = mean((y_test .- regressors_test*blr_coeffs).^2)




"""
Estimate 2 stage BLR then supervised LDA (sLDA) on residuals
"""
## Residualise first
blr_coeffs_post, σ2_post = BLR_Gibbs(y_train,
    hcat(ones(size(x_train,1)),x_train), iteration = M_iteration,
    m_0 = μ_ω, V_0 = σ_ω, a_0 = a_0, b_0 = b_0)
blr_coeffs = Array{Float64,1}(vec(mean(blr_coeffs_post, dims = 2)))
resids_blr_train = y_train - hcat(ones(size(x_train,1)),x_train)*blr_coeffs

ω_kk = sort(blr_coeffs_post[2,:])
lo = ω_kk[Int(round(0.025*M_iteration))]
hi = ω_kk[Int(round(0.975*M_iteration))]
mid = mean(ω_kk)


## Estimate sLDA on residuals
@time results_blr_slda =
    BTR_EMGibbs(dtm_train, ntopics, resids_blr_train,
        α = α, η = η, σ_ω = σ_ω, a_0 = a_0, b_0 = b_0,
        batch = true, ω_tol = ω_tol,
        E_iteration = E_iteration, EM_iteration = EM_iteration,
        M_iteration = M_iteration, EM_split = EM_split, plot_ω=true)

## Plot results
topic_order = synth_reorder_topics(results_blr_slda.β)
# Without interactions
plt = synth_data_plot(results_blr_slda.β, results_blr_slda.ω_post, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(size = (300,300))
savefig("figures/synth_LR_sLDA.pdf")


## Out of sample prediction
y_stage1_test = hcat(ones(size(x_test,1)),x_test)*blr_coeffs
resids_test = y_test - y_stage1_test

predict_blr_slda =
    BTR_Gibbs_predict(dtm_test, ntopics, results_blr_slda.β, results_blr_slda.ω,
    α = α, Σ = results_blr_slda.Σ, E_iteration = E_iteration, burnin = burnin)
mse_blr_slda = mean((y_test .- (predict_blr_slda.y_pred .+ y_stage1_test)).^2)




"""
Estimate 2 stage slDA then BLR on residuals
"""
## Estimate sLDA on residuals
@time results_slda =
    BTR_EMGibbs(dtm_train, ntopics, y_train,
        α = α, η = η, σ_ω = σ_ω, a_0 = a_0, b_0 = b_0,
        batch = true, ω_tol = ω_tol,
        E_iteration = E_iteration, EM_iteration = EM_iteration,
        M_iteration = M_iteration, EM_split = EM_split, plot_ω=true)

## Plot results
topic_order = synth_reorder_topics(results_slda.β)
# Without interactions
plt = synth_data_plot(results_slda.β, results_slda.ω_post, true_ω = ω_z_true,
    topic_ord = topic_order, plt_title = "",
    left_mar = 3,top_mar = 0, bottom_mar = 0, ticksize = 10, labelsize = 25)
plot!(size = (300,300))
savefig("figures/synth_sLDA_LR.pdf")


## Identify residuals to train second stage regression
slda_nobatch_y = BTR_Gibbs_predict(dtm_train, ntopics, results_slda.β, results_slda.ω,
    α = α, Σ = results_slda.Σ,
    E_iteration = E_iteration, burnin = burnin)
residuals_slda = y_train - slda_nobatch_y.y_pred


## Bayesian linear regression on training set
# Create regressors
regressors_train = hcat(ones(size(x_train,1)),x_train)

# No fixed effect, no batch
blr_coeffs_post, σ2_post = BLR_Gibbs(residuals_slda,
    regressors_train, iteration = M_iteration,
    m_0 = μ_ω, V_0 = σ_ω, a_0 = a_0, b_0 = b_0)
blr_coeffs = Array{Float64,1}(vec(mean(blr_coeffs_post, dims = 2)))

ω_kk = sort(blr_coeffs_post[2,:])
lo = ω_kk[Int(round(0.025*M_iteration))]
hi = ω_kk[Int(round(0.975*M_iteration))]
mid = mean(ω_kk)


## Out of sample
predict_slda =
    BTR_Gibbs_predict(dtm_test, ntopics, results_slda.β, results_slda.ω,
        α = α, Σ = results_slda.Σ,
        E_iteration = E_iteration, burnin = burnin)

# Second stage regressors
regressors_test = hcat(ones(size(x_test,1)),x_test)

# Add y prediction from slda to portion of residual explained by BLR
y_pred_slda_blr = predict_slda.y_pred + regressors_test*blr_coeffs

# Compute MSE
mse_slda_blr = mean((y_test .- y_pred_slda_blr).^2)






"""
Compare out of sample performance for each model
"""
## Restrict range so that figure isn't too busy
plot_range = 1:50
# True data
plot(y_test[plot_range], linestyle = :solid,
    label = join(["True data"]), title = "")
# BTR with interactions
plot!(predict_btr.y_pred[plot_range], linestyle = :solid,
    label = join(["BTR (", round(mse_btr,digits = 3), ")"]))
# BLR then SLDA
plot!((predict_slda.y_pred .+ y_stage1_test)[plot_range], linestyle = :dashdot,
    label = join(["LR + sLDA (", round(mse_blr_slda,digits = 3), ")"]))
# BLR then SLDA
plot!((predict_blr_slda.y_pred .+ y_stage1_test)[plot_range], linestyle = :dot,
    label = join(["sLDA + LR (", round(mse_slda_blr,digits = 3), ")"]))
# BlR without documents
plot!(predict_blr[plot_range], linestyle = :dash,
    label = join(["BLR (", round(mse_blr,digits = 3), ")"]))
savefig("figures/synth_mse_comparison.pdf")
