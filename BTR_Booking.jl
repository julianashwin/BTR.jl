"""
Implementation of BTR with Gibbs sampling for Booking dataset
"""

cd("/Users/julianashwin/Documents/GitHub/BTR.jl/")

"""
To install the package is run, enter pkg mode by running "]" then run
pkg> dev path_to_folder/BTR.jl
"""

"""
Load data and necessary packages
"""
## Packages
using BTR
using TextAnalysis, DataFrames, CSV, Random, GLM, Distributions
using Plots, StatsPlots, StatsBase, Plots.PlotMeasures, TableView

## Load data
df = CSV.read("data/Booking_sample.csv", DataFrame, threaded = false)
# Check that the variables look sensible
#display(plot(df.date[1:200], df.stars[1:200], label = ["stars"], legend = :bottomleft,xguidefontsize=8))

## Toggle whether to save the various figures output throughout
save_files = false


"""
Create document ids from either review id or business id
"""
## The ids should be in Int64 format

#for doc level
df[!,:doc_idx] = 1:nrow(df) # for doc-level

# for para-level
#df.business_id = string.(df.business_id)
#df[!,:doc_idx] = convert_to_ids(df.business_id)
#sort!(df, [:doc_idx])
#showtable(df)




"""
Prepare data for estimation
"""
## Create labels and covariates

x = Array{Float64,2}(hcat(df.Average_Score,
        df.Review_Total_Negative_Word_Counts,
        df.Total_Number_of_Reviews,
        df.Review_Total_Positive_Word_Counts,
        df.Total_Number_of_Reviews_Reviewer_Has_Given))

y = Array{Float64,1}(df.Reviewer_Score)



"
x = group_mean(Array{Float64,2}(hcat(df.sentiment,df.stars_av_u)),df.doc_idx)
y = group_mean(Array{Float64,1}(df.stars_av_b), df.doc_idx)
"
docidx_vars = unique(df.doc_idx)
docidx_dtm = df.doc_idx
D = length(unique(docidx_dtm))


## Use TextAnalysis package to create a DTM
docs = StringDocument.(df.Full_Review_Clean)
crps = Corpus(docs)
update_lexicon!(crps)
dtm_sparse = DocumentTermMatrix(crps)
vocab = dtm_sparse.terms


"""
Split into training and test sets (by doc_idx) and convert to BTRRawData structure
"""
## Extract sparse matrix and create BTRRawData structure(s)
dtm_in = dtm_sparse.dtm
train_data, test_data = btr_traintestsplit(dtm_in, docidx_dtm, docidx_vars, y, vocab, x = x,
    train_split = 0.75, shuffle_obs = false)
# Alternatively, can convert the entier set to BTRRawData with
all_data = DocStructs.BTRRawData(dtm_in, docidx_dtm, docidx_vars, y, x, vocab)
## Visualise the training-test split
histogram(train_data.docidx_dtm, bins = 1:D, label = "training set",
    xlab = "Observation", ylab= "Paragraphs", c=1, lc=nothing)
histogram!(test_data.docidx_dtm, bins = 1:D, label = "test set", c=2, lc=nothing)
if save_files; savefig("figures/Booking_BTR/Booking_trainsplit.pdf"); end;



"""
Standardise using only the training data
"""
## Use mean and std from training data to normalise both sets
y_mean_tr = mean(train_data.y)
y_std_tr = std(train_data.y)
x_mean_tr = mean(train_data.x,dims=1)
x_std_tr = std(train_data.x,dims=1)
train_data.y = (train_data.y .- y_mean_tr)#./y_std_tr
train_data.x = (train_data.x .- x_mean_tr)./x_std_tr
test_data.y = (test_data.y .- y_mean_tr)#./y_std_tr
test_data.x = (test_data.x .- x_mean_tr)./x_std_tr




"""
Set priors and estimation optioncs here to be consistent across models
"""
## Initialiase estimation options
btropts = BTROptions()
## Number of topics
btropts.ntopics = 100
## LDA priors
btropts.α=0.5
btropts.η=0.01
## BLR priors
btropts.μ_ω = 0. # coefficient mean
btropts.σ_ω = 2. # coefficient variance
btropts.a_0 = 3. # residual shape: higher moves mean closer to zero
btropts.b_0 = 2. # residual scale: higher is more spread out
# Plot the prior distribution for residual variance (in case unfamiliar with InverseGamma distributions)
# mean will be b_0/(a_0 - 1)
plot(InverseGamma(btropts.a_0, btropts.b_0), xlim = (0,3), title = "Residual variance prior",
    label = "Prior on residual variance")
scatter!([var(y)],[0.],label = "Unconditional variance")
if save_files; savefig("figures/Booking_BTR/Booking_IGprior.pdf"); end;

## Number of iterations and convergence tolerance
btropts.E_iters = 100 # E-step iterations (sampling topic assignments, z)
btropts.M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
btropts.EM_iters = 50 # Maximum possible EM iterations (will stop here if no convergence)
btropts.CVEM = :obs # Split for separate E and M step batches (if batch = true)
btropts.CVEM_split = 0.5 # Split for separate E and M step batches (if batch = true)
btropts.burnin = 20 # Burnin for Gibbs samplers

btropts.mse_conv = 1
btropts.ω_tol = 0.01 # Convergence tolerance for regression coefficients ω
btropts.rel_tol = true # Whether to use a relative convergence criteria rather than just absolute


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
## Out-of-sample
predict_blr = regressors_test*blr_coeffs
mse_blr = mean((test_data.y .- predict_blr).^2)



"""
Estimate BTR
"""
## Include x regressors by changing the options
btropts.xregs = [1,2,3,4,5]
btropts.interactions = Array{Int64,1}([])
## Initialise BTRModel object
btrcrps_tr = create_btrcrps(train_data, btropts.ntopics)
btrcrps_ts = create_btrcrps(test_data, btropts.ntopics)
btrmodel = BTRModel(crps = btrcrps_tr, options = btropts)
## Estimate BTR with EM-Gibbs algorithm
btropts.CVEM = :obs
btropts.CVEM_split = 0.5
btrmodel = BTRemGibbs(btrmodel)
## Plot results
BTR_plot(btrmodel.β, btrmodel.ω_post, btrmodel.crps.vocab,
    plt_title = "Booking BTR", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_BTR.pdf"); end;

## Out of sample prediction in test set
btr_predicts = BTRpredict(btrcrps_ts, btrmodel)
#btr_predicts.crps.topics = gettopics(btrcrps_ts.docs)
mse_btr = mean((btr_predicts.y .- btr_predicts.y_pred).^2)
pplxy_btr = btr_predicts.pplxy

"""
Estimate 2 stage LDA then Bayesian Linear Regression (BLR)
    In the synthetic data this does about as well as BTR because the
    text is generated from an LDA model.
"""
## Use the same options as the BTR, but might want to tweak the number of iterations as there's only one step
ldaopts = deepcopy(btropts)
ldaopts.fullGibbs_iters = 1000
ldaopts.fullGibbs_thinning = 2
ldaopts.burnin = 50
## Initialise model (re-initialise the corpora to start with randomised assignments)
ldacrps_tr = create_btrcrps(train_data, ldaopts.ntopics)
ldacrps_ts = create_btrcrps(test_data, ldaopts.ntopics)
ldamodel = BTRModel(crps = ldacrps_tr, options = ldaopts)
## Estimate LDA model on full training set
ldamodel  = LDAGibbs(ldamodel)

## Bayesian linear regression on training set
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(ldamodel.y, ldamodel.regressors,
    m_0 = ldaopts.μ_ω, σ_ω = ldaopts.σ_ω, a_0 = ldaopts.a_0, b_0 = ldaopts.b_0)
ldamodel.ω = blr_ω
ldamodel.ω_post = blr_ω_post

## Plot results
BTR_plot(ldamodel.β, ldamodel.ω_post, ldamodel.crps.vocab,
    plt_title = "Booking LDA", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_LDA.pdf"); end;

## Out of sample prediction
lda_predicts = BTRpredict(ldacrps_ts, ldamodel)
## Calculate MSE
mse_lda_blr = mean((lda_predicts.y .- lda_predicts.y_pred).^2)
pplxy_lda_blr = lda_predicts.pplxy


"""
Estimate 2 stage BLR then supervised LDA (sLDA) on residuals
"""
## Residualise first
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(train_data.y, hcat(ones(size(train_data.x,1)),train_data.x),
    m_0 = ldaopts.μ_ω, σ_ω = ldaopts.σ_ω, a_0 = ldaopts.a_0, b_0 = ldaopts.b_0)
# Get the y residualised on x
resids_blr_train = train_data.y - hcat(ones(size(train_data.x,1)),train_data.x)*blr_ω

## Set options sLDA on residuals
slda1opts = deepcopy(btropts)
slda1opts.xregs = []
slda1opts.interactions = []

## Initialise BTRModel object
slda1crps_tr = create_btrcrps(train_data, btropts.ntopics)
slda1crps_ts = create_btrcrps(test_data, btropts.ntopics)
for ii in 1:slda1crps_tr.N
    slda1crps_tr.docs[ii].y = resids_blr_train[ii]
end
slda1model = BTRModel(crps = slda1crps_tr, options = slda1opts)

## Estimate sLDA on residuals
slda1model = BTRemGibbs(slda1model)

## Plot results
BTR_plot(slda1model.β, slda1model.ω_post, slda1model.crps.vocab,
    plt_title = "Booking BLR + sLDA", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_LR_sLDA.pdf"); end;

## Out of sample prediction
y_stage1_test = hcat(ones(size(test_data.x,1)),test_data.x)*blr_ω
resids_test = test_data.y - y_stage1_test
# Create corpus with residual as y
for ii in 1:slda1crps_ts.N
    slda1crps_ts.docs[ii].y = resids_test[ii]
end
slda1_predicts = BTRpredict(slda1crps_ts, slda1model)
# Performance
mse_blr_slda = mean((slda1_predicts.y .-slda1_predicts.y_pred).^2)
pplxy_blr_slda = slda1_predicts.pplxy



"""
Estimate 2 stage slDA then BLR on residuals
"""
## Set options sLDA on residuals
slda2opts = deepcopy(btropts)
slda2opts.xregs = []
slda2opts.interactions = []

## Initialise BTRModel object
slda2crps_tr = create_btrcrps(train_data, slda2opts.ntopics)
slda2crps_ts = create_btrcrps(test_data, slda2opts.ntopics)
slda2model = BTRModel(crps = slda2crps_tr, options = slda2opts)

## Estimate sLDA on residuals
slda2model = BTRemGibbs(slda2model)

## Plot results
BTR_plot(slda2model.β, slda2model.ω_post, slda2model.crps.vocab,
    plt_title = "Booking sLDA + BLR", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_sLDA_LR.pdf"); end;

## Identify residuals to train second stage regression
residuals_slda = train_data.y .- slda2model.regressors*slda2model.ω

## Bayesian linear regression on training set
# Create regressors
regressors_slda = hcat(ones(size(train_data.x,1)),train_data.x)
# No fixed effect, no batch
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(residuals_slda, regressors_slda,
    m_0 = slda2opts.μ_ω, σ_ω = slda2opts.σ_ω, a_0 = slda2opts.a_0, b_0 = slda2opts.b_0)

## Out of sample
slda2_predicts = BTRpredict(slda2crps_ts, slda2model)
# Second stage regressors
regressors_test = hcat(ones(size(test_data.x,1)),test_data.x)
# Add y prediction from slda to portion of residual explained by BLR
y_pred_slda_blr = slda2_predicts.y_pred + regressors_test*blr_ω
# Compute MSE
mse_slda_blr = mean((test_data.y .- y_pred_slda_blr).^2)



"""
Multiple runs
"""
## Set subdirectory and number of times you want to run
subdirectory = "/Users/julianashwin/Desktop/BTR_runs/Booking/run_"
nruns = 2
## Run multiple times (for different hyperparameters change btropts)
BTR_multipleruns(train_data, test_data, btropts, nruns, subdirectory)




"""
End of script
"""
