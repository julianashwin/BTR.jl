"""
Implementation of BTR with Gibbs sampling for Yelp dataset
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
df = CSV.read("data/yelp_toronto_sample.csv", DataFrame, threaded = false)
# Check that the variables look sensible
display(plot(df.stars[1:200], label = ["stars"], legend = :bottomleft,xguidefontsize=8))

## Toggle whether to save the various figures output throughout
save_files = false


"""
Create document ids from either review id or business id
"""
## The ids should be in Int64 format
df.review_id = string.(df.review_id)
df.doc_idx = 1:nrow(df)
#df[!,:doc_idx] = convert_to_ids(df.review_id)
#sort!(df, [:doc_idx])
#showtable(df)


"""
Process text
"""
## Use TextAnalysis package to create a DTM
docs = StringDocument.(df.text_clean)
crps = Corpus(docs)
update_lexicon!(crps)

lex_df = DataFrame(term = String.(keys(lexicon(crps))),
    count = Int.(values(lexicon(crps))))
vocab = lex_df.term[(lex_df.count .> 10)]
dtm_sparse = DocumentTermMatrix(crps, vocab)
dtm_sparse.terms


nwords = vec(sum(dtm_sparse.dtm, dims = 2))
pos_counts = Float64.(wordlistcounts(dtm_sparse.dtm,vocab,HIV_dicts.Positive))
neg_counts = Float64.(wordlistcounts(dtm_sparse.dtm,vocab,HIV_dicts.Negative))
df[:, "harvard_score"] .= (pos_counts.-neg_counts)./nwords
df.harvard_score[isnan.(df.harvard_score)] .= 0.
df.harvard_score = (df.harvard_score.-mean(df.harvard_score))./std(df.harvard_score)

"""
Create synthetic variables
"""
## Set γ_1 for strength of correlation with confounder
γ_1 = 1.0
γ_0 = 0.0
σ_y_true = 0.1

df[:,"US"] .= 0.
df[:,"PrUS"] .= exp.(γ_0 .+ γ_1.*df.harvard_score)./(1.0 .+ exp.(γ_0 .+ γ_1.*df.harvard_score))

df[:,"US"] .= Float64.(rand.(Binomial.(1,df.PrUS)))

mean(df.US)
display(cor(hcat(df.US, df.PrUS, df.sentiment)))

df[:,"synth_y"] = 1.0.*df.stars_av_b + 1.0.*df.harvard_score - 1.0 .* df.US + 0.1.*randn(nrow(df))



"""
No text regressions
"""
## Full
fm = @formula(synth_y ~ stars_av_b + US + harvard_score)
synth_full = lm(fm, df)
display(synth_full)
## No text
fm = @formula(synth_y ~ stars_av_b + US)
synth_notext = lm(fm, df)
display(synth_notext)
## No text or other xs
fm = @formula(synth_y ~ US)
synth_nox = lm(fm, df)
display(synth_nox)


"""
Prepare data for estimation
"""
## Create labels and covariates
x = group_mean(Array{Float64,2}(hcat(df.stars_av_b, df.US,)),df.doc_idx)
y = group_mean(Array{Float64,1}(df.synth_y), df.doc_idx)
docidx_vars = df.doc_idx
docidx_dtm = df.doc_idx
D = length(unique(docidx_dtm))


"""
Split into training and test sets (by doc_idx) and convert to BTRRawData structure
"""
## Extract sparse matrix and create BTRRawData structure(s)
dtm_in = dtm_sparse.dtm
train_data, test_data = btr_traintestsplit(dtm_in, docidx_dtm, docidx_vars, y, vocab, x = x,
    train_split = 1.0, shuffle_obs = false)
# Alternatively, can convert the entier set to BTRRawData with
all_data = DocStructs.BTRRawData(dtm_in, docidx_dtm, docidx_vars, y, x, vocab)
## Visualise the training-test split
histogram(train_data.docidx_dtm, bins = 1:D, label = "training set",
    xlab = "Observation", ylab= "Paragraphs", c=1, lc=nothing)
histogram!(test_data.docidx_dtm, bins = 1:D, label = "test set", c=2, lc=nothing)
if save_files; savefig("figures/Yelp_BTR/Yelp_trainsplit.pdf"); end;


"""
Standardise using only the training data
"""
## Use mean and std from training data to normalise both sets
#y_mean_tr = mean(train_data.y)
#y_std_tr = std(train_data.y)
#x_mean_tr = mean(train_data.x,dims=1)
#x_std_tr = std(train_data.x,dims=1)
#train_data.y = (train_data.y .- y_mean_tr)#./y_std_tr
#train_data.x = (train_data.x .- x_mean_tr)./x_std_tr
#test_data.y = (test_data.y .- y_mean_tr)#./y_std_tr
#test_data.x = (test_data.x .- x_mean_tr)./x_std_tr



"""
Set priors and estimation optioncs here to be consistent across models
"""
## Initialiase estimation options
opts = BTROptions()
## Number of topics
opts.ntopics = 10
## LDA priors
opts.α=0.5
opts.η=0.1
## BLR priors
opts.μ_ω = 0. # coefficient mean
opts.σ_ω = 2. # coefficient variance
opts.a_0 = 4. # residual shape: higher moves mean closer to zero
opts.b_0 = 2. # residual scale: higher is more spread out
# Plot the prior distribution for residual variance (in case unfamiliar with InverseGamma distributions)
# mean will be b_0/(a_0 - 1)
plot(InverseGamma(opts.a_0, opts.b_0), xlim = (0,2.), title = "Residual variance prior",
    label = "Prior on residual variance")
scatter!([var(train_data.y)],[0.],label = "Unconditional variance")
if save_files; savefig("figures/Yelp_BTR/Yelp_IGprior.pdf"); end;

## Number of iterations and cross validation
opts.E_iters = 100 # E-step iterations (sampling topic assignments, z)
opts.M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
opts.EM_iters = 25 # Maximum possible EM iterations (will stop here if no convergence)
opts.burnin = 10 # Burnin for Gibbs samplers
opts.CVEM = :obs # Split for separate E and M step batches (if batch = true)
opts.CVEM_split = 0.5 # Split for separate E and M step batches (if batch = true)

## Comvergence options
opts.mse_conv = 1
opts.ω_tol = 0.015 # Convergence tolerance for regression coefficients ω
opts.rel_tol = true # Whether to use a relative convergence criteria rather than just absolute

opts.xregs = [1,2]
opts.interactions = Array{Int64,1}([])


"""
Run some text-free regressions for benchmarking
"""
## Define regressors
regressors_notext = hcat(ones(size(all_data.x,1)),all_data.x)
## OLS
ols_coeffs = inv(transpose(regressors_notext)*regressors_notext)*(transpose(regressors_notext)*all_data.y)
## Bayesian linear regression
blr_coeffs, blr_σ2, blr_notext_coeffs_post, σ2_post = BLR_Gibbs(all_data.y, regressors_notext,
    iteration = opts.M_iters, m_0 = opts.μ_ω, σ_ω = opts.σ_ω, a_0 = opts.a_0, b_0 = opts.b_0)
blr_coeffs = Array{Float64,1}(vec(mean(blr_notext_coeffs_post, dims = 2)))

notext_TE = blr_coeffs[3]

TE_post_df = DataFrame(NoText_reg = sort(blr_notext_coeffs_post[3,:]))




"""
Estimate BTR without CVEM
"""
## Options without CVEM
btropts_noCVEM = deepcopy(opts)
btropts_noCVEM.CVEM = :none
btropts_noCVEM.CVEM_split = 0.5
## Include x regressors by changing the options
btropts_noCVEM.xregs = [1,2]
btropts_noCVEM.interactions = Array{Int64,1}([])
## Initialise BTRModel object
btrcrps_tr = create_btrcrps(all_data, btropts_noCVEM.ntopics)
btrmodel_noCVEM = BTRModel(crps = btrcrps_tr, options = btropts_noCVEM)
## Estimate BTR with EM-Gibbs algorithm
btrmodel_noCVEM = BTRemGibbs(btrmodel_noCVEM)



## Plot results
BTR_plot(btrmodel_noCVEM.β, btrmodel_noCVEM.ω_post, btrmodel_noCVEM.crps.vocab,
    plt_title = "Yelp synth BTR (No CVEM)", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_BTR.pdf"); end;

btr_noCEVM_TE = btrmodel_noCVEM.ω[btropts_noCVEM.ntopics+2]
btr_pplxy = btrmodel_noCVEM.pplxy

TE_post_df[:,"BTR_noCVEM"] = sort(btrmodel_noCVEM.ω_post[btropts_noCVEM.ntopics+2,:])



"""
Estimate BTR with CVEM
"""
## Options without CVEM
btropts_CVEM = deepcopy(opts)
btropts_CVEM.CVEM = :obs
btropts_CVEM.CVEM_split = 0.5
## Include x regressors by changing the options
btropts_CVEM.xregs = [1,2]
btropts_CVEM.interactions = Array{Int64,1}([])
## Initialise BTRModel object
btrcrps_tr = create_btrcrps(all_data, btropts_CVEM.ntopics)
btrmodel_CVEM = BTRModel(crps = btrcrps_tr, options = btropts_CVEM)
## Estimate BTR with EM-Gibbs algorithm
btrmodel_CVEM = BTRemGibbs(btrmodel_CVEM)



## Plot results
BTR_plot(btrmodel_CVEM.β, btrmodel_CVEM.ω_post, btrmodel_CVEM.crps.vocab,
    plt_title = "Yelp synth BTR (CVEM)", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_BTR.pdf"); end;

btr_CVEM_TE = btrmodel_CVEM.ω[btropts_CVEM.ntopics+2]
btr_pplxy = btrmodel_CVEM.pplxy

TE_post_df[:,"BTR_CVEM"] = sort(btrmodel_CVEM.ω_post[btropts_CVEM.ntopics+2,:])

"""
Repeat for many K
"""
TE_Krobustness_df = DataFrame(NoText_reg = sort(blr_notext_coeffs_post[3,:]))
for K in [2,3,4,5,6,7,8,9,10,15,20,15,30,40,50]
    display("Estimating with "*string(K)*" topics")
    btropts_CVEM = deepcopy(opts)
    btropts_CVEM.ntopics = K

    btrcrps_tr = create_btrcrps(all_data, btropts_CVEM.ntopics)
    btrmodel_CVEM = BTRModel(crps = btrcrps_tr, options = btropts_CVEM)
    ## Estimate BTR with EM-Gibbs algorithm
    btrmodel_CVEM = BTRemGibbs(btrmodel_CVEM)

    TE_Krobustness_df[:,"BTR_CVEM_K"*string(K)] = sort(btrmodel_CVEM.ω_post[btropts_CVEM.ntopics+2,:])
end




"""
Estimate 2 stage LDA then Bayesian Linear Regression (BLR)
    In the synthetic data this does about as well as BTR because the
    text is generated from an LDA model.
"""
## Use the same options as the BTR, but might want to tweak the number of iterations as there's only one step
ldaopts = deepcopy(opts)
ldaopts.fullGibbs_iters = 1000
ldaopts.fullGibbs_thinning = 2
ldaopts.burnin = 50
## Initialise model (re-initialise the corpora to start with randomised assignments)
ldacrps_tr = create_btrcrps(all_data, ldaopts.ntopics)
ldamodel = BTRModel(crps = ldacrps_tr, options = ldaopts)
## Estimate LDA model on full training set
ldamodel  = LDAGibbs(ldamodel)

## Bayesian linear regression on training set
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(ldamodel.y, ldamodel.regressors,
    m_0 = ldaopts.μ_ω, σ_ω = ldaopts.σ_ω, a_0 = ldaopts.a_0, b_0 = ldaopts.b_0,
    iteration = ldaopts.M_iters)
ldamodel.ω = blr_ω
ldamodel.ω_post = blr_ω_post

lda_TE = ldamodel.ω[ldaopts.ntopics+2]

## Plot results
BTR_plot(ldamodel.β, ldamodel.ω_post, ldamodel.crps.vocab,
    plt_title = "Yelp synth LDA", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_LDA.pdf"); end;


TE_post_df[:,"LDA"] = sort(ldamodel.ω_post[ldaopts.ntopics+2,:])




"""
Estimate 2 stage slDA then BLR on residuals
"""
## Set options sLDA on residuals
slda2opts = deepcopy(opts)
slda2opts.xregs = Array{Int64,1}([])
slda2opts.interactions = Array{Int64,1}([])

## Initialise BTRModel object
slda2crps_tr = create_btrcrps(all_data, slda2opts.ntopics)
slda2model = BTRModel(crps = slda2crps_tr, options = slda2opts)

## Estimate sLDA on residuals
slda2model = BTRemGibbs(slda2model)

## Plot results
BTR_plot(slda2model.β, slda2model.ω_post, slda2model.crps.vocab,
    plt_title = "Yelp synth sLDA + BLR", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_sLDA_LR.pdf"); end;

## Identify residuals to train second stage regression
residuals_slda = train_data.y .- slda2model.regressors*slda2model.ω

## Bayesian linear regression on training set
# Create regressors
regressors_slda = hcat(ones(size(train_data.x,1)),train_data.x)
# No fixed effect, no batch
blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(residuals_slda, regressors_slda,
    m_0 = slda2opts.μ_ω, σ_ω = slda2opts.σ_ω, a_0 = slda2opts.a_0, b_0 = slda2opts.b_0,
    iteration = slda2opts.M_iters)

slda2_TE = blr_ω[3]

TE_post_df[:,"sLDA"] = sort(blr_ω_post[3,:])




"""
Plot treatment effects
"""
nmodels = 5
model_names = ["BTR (no CVEM)","BTR (CVEM)", "LDA", "sLDA", "NoText BLR"]
plt1 = plot(legend = false, ylim = (0,nmodels+1), xlim = (0., 2.0),
    xlabel = "Treatment Effect", ylabel = "Model",
    yticks = (1:nmodels, model_names))
plot!([1.,1.],[0.,(Float64(nmodels)+0.6)],linestyle = :dash,color =:red)

coef_plot(sort(TE_post_df.BTR_noCVEM),1,scheme = :blue)
coef_plot(sort(TE_post_df.BTR_CVEM),2,scheme = :blue)
coef_plot(sort(TE_post_df.LDA),3,scheme = :blue)
coef_plot(sort(TE_post_df.sLDA),4,scheme = :blue)
coef_plot(sort(TE_post_df.NoText_reg),5,scheme = :blue)



"""
End of script
"""
