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
df = CSV.read("data/booking_semisynth_sample.csv", DataFrame, threaded = false)
# Check that the variables look sensible
#display(plot(df.date[1:200], df.stars[1:200], label = ["stars"], legend = :bottomleft,xguidefontsize=8))

## Toggle whether to save the various figures output throughout
save_files = false
regenerate_data = false

"""
Empirical regression
"""
## Regression on true reviewer score
fm = @formula(Reviewer_Score ~ av_score + Leisure + Couple + pos_prop)
empirical_lm = lm(fm, df)
display(empirical_lm)


"""
Create DTM from cleaned text
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

sort!(lex_df, :count, rev = true)
showtable(lex_df)

"""
Re-generate the semi-synthetic data if this is toggled to true
"""

if regenerate_data
    tf_df = DataFrame(id = 1:length(vocab), term = vocab, freq = vec(sum(dtm_sparse.dtm, dims=1)))
    sort!(tf_df, :freq, rev= true)

    ##  Create potential covariates
    df.text = string.(df.text)
    df.sentiment = sentimentscore(df.text_clean, HIV_dicts)
    df[:,"pos_prop"] .= df.Review_Total_Positive_Word_Counts./(
        df.Review_Total_Negative_Word_Counts .+ df.Review_Total_Positive_Word_Counts)
    df[:,"av_score"] = (df.Average_Score .- mean(df.Average_Score))./std(df.Average_Score)

    ## Word counts
    #nwords = vec(sum(dtm_sparse.dtm, dims = 2))
    #pos_list = ["great","good","excel","love","nice"]
    #work_list = ["work","confer","busi","convent"]
    ## Get wordcounts
    # Positive
    pos_list = HIV_dicts.Positive
    pos_counts = Float64.(wordlistcounts(dtm_sparse.dtm,vocab,pos_list))
    df[:, "pos_score"] .= pos_counts./nwords
    df.pos_score[isnan.(df.pos_score)] .= 0.
    df.pos_score = (df.pos_score.-mean(df.pos_score))./std(df.pos_score)
    # Work
    #work_list = ["work","confer","busi","convent"]
    #work_counts = Float64.(wordlistcounts(dtm_sparse.dtm,vocab,pos_list))
    #df[:, "work_score"] .= work_counts./nwords
    #df.work_score[isnan.(df.work_score)] .= 0.

    ## Create synthetic target
    df.synth_y = 1.0.*df.av_score + 1.0.*df.Leisure +
        10.0.*df.pos_prop + 0.1.*randn(nrow(df))

    display(cor(hcat(df.Leisure, df.pos_prop, df.pos_score, nwords)))

    ## Export
    #CSV.write("data/booking_semisynth_sample.csv", df)
end

display(cor(hcat(df.Leisure, df.sentiment, df.pos_prop, df.synth_y)))





"""
No text regressions
"""
## No text
fm = @formula(synth_y ~ av_score + Leisure)
synth_notext = lm(fm, df)
display(synth_notext)
## No text or other xs
fm = @formula(synth_y ~ Leisure)
synth_nox = lm(fm, df)
display(synth_nox)


"""
Prepare data for estimation
"""
## Documents ids
df[!,:doc_idx] = 1:nrow(df) # for doc-level
docidx_vars = unique(df.doc_idx)
docidx_dtm = df.doc_idx
D = length(unique(docidx_dtm))

## Create labels and covariates
x = Array{Float64,2}(hcat(df.av_score,df.Leisure))
y = Array{Float64,1}(df.synth_y)


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
if save_files; savefig("figures/Booking_BTR/Booking_trainsplit.pdf"); end;




"""
Set priors and estimation optioncs here to be consistent across models
"""
## Initialiase estimation options
opts = BTROptions()
## Number of topics
opts.ntopics = 20
## LDA priors
opts.α=0.5
opts.η=0.01
## BLR priors
opts.μ_ω = 0. # coefficient mean
opts.σ_ω = 2. # coefficient variance
opts.a_0 = 3. # residual shape: higher moves mean closer to zero
opts.b_0 = 2. # residual scale: higher is more spread out
# Plot the prior distribution for residual variance (in case unfamiliar with InverseGamma distributions)
# mean will be b_0/(a_0 - 1)
plot(InverseGamma(opts.a_0, opts.b_0), xlim = (0,3), title = "Residual variance prior",
    label = "Prior on residual variance")
scatter!([var(y)],[0.],label = "Unconditional variance")
if save_files; savefig("figures/Booking_BTR/Booking_IGprior.pdf"); end;

## Number of iterations and convergence tolerance
opts.E_iters = 100 # E-step iterations (sampling topic assignments, z)
opts.M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
opts.EM_iters = 50 # Maximum possible EM iterations (will stop here if no convergence)
opts.CVEM = :obs # Split for separate E and M step batches (if batch = true)
opts.CVEM_split = 0.5 # Split for separate E and M step batches (if batch = true)
opts.burnin = 20 # Burnin for Gibbs samplers

opts.mse_conv = 2
opts.ω_tol = 0.01 # Convergence tolerance for regression coefficients ω
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
blr_coeffs, blr_σ2, blr_notext_coeffs_post, σ2_post = BLR_Gibbs(all_data.y, regressors_notext, iteration = opts.M_iters,
    m_0 = opts.μ_ω, σ_ω = opts.σ_ω, a_0 = opts.a_0, b_0 = opts.b_0)
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
    plt_title = "Booking BTR (No CVEM)", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_BTR.pdf"); end;

btr_TE = btrmodel_noCVEM.ω[btropts_noCVEM.ntopics+2]
btr_pplxy = btrmodel_noCVEM.pplxy

TE_post_df[:,"BTR_noCVEM"] = sort(btrmodel_noCVEM.ω_post[btropts_noCVEM.ntopics+2,:])


## Repeat for many K
TE_Krobustness_df = DataFrame(NoText_reg = sort(blr_notext_coeffs_post[3,:]))
Ks = [2,3,4,5,6,7,8,9,10,15,20,25,30,40,50]
for K in Ks
    display("Estimating with "*string(K)*" topics")
    btropts_noCVEM = deepcopy(opts)
    btropts_noCVEM.CVEM = :none
    btropts_noCVEM.ntopics = K
    btropts_noCVEM.mse_conv = 2

    btrcrps_tr = create_btrcrps(all_data, btropts_noCVEM.ntopics)
    btrmodel_noCVEM = BTRModel(crps = btrcrps_tr, options = btropts_noCVEM)
    ## Estimate BTR with EM-Gibbs algorithm
    btrmodel_noCVEM = BTRemGibbs(btrmodel_noCVEM)

    TE_Krobustness_df[:,"BTR_noCVEM_K"*string(K)] = sort(btrmodel_noCVEM.ω_post[btropts_noCVEM.ntopics+2,:])
end

mean_TEs = vec(mean(Matrix(TE_Krobustness_df[:,(2:(length(Ks)+1))]), dims = 1))
plot(Ks, mean_TEs, title = "Booking semi-synth, without CVEM")


"""
Estimate BTR with CVEM
"""
## Options without CVEM
btropts_CVEM = deepcopy(opts)
btropts_CVEM.CVEM = :obs
btropts_CVEM.CVEM_split = 0.5
btropts_CVEM.mse_conv = 2
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
    plt_title = "Booking BTR (CVEM)", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_BTR.pdf"); end;

btr_TE = btrmodel_CVEM.ω[btropts_CVEM.ntopics+2]
btr_pplxy = btrmodel_CVEM.pplxy

TE_post_df[:,"BTR_CVEM"] = sort(btrmodel_CVEM.ω_post[btropts_CVEM.ntopics+2,:])



## Repeat for many K
for K in Ks
    display("Estimating with "*string(K)*" topics")
    btropts_CVEM = deepcopy(opts)
    btropts_CVEM.ntopics = K
    btropts_CVEM.mse_conv = 2

    btrcrps_tr = create_btrcrps(all_data, btropts_CVEM.ntopics)
    btrmodel_CVEM = BTRModel(crps = btrcrps_tr, options = btropts_CVEM)
    ## Estimate BTR with EM-Gibbs algorithm
    btrmodel_CVEM = BTRemGibbs(btrmodel_CVEM)

    TE_Krobustness_df[:,"BTR_CVEM_K"*string(K)] = sort(btrmodel_CVEM.ω_post[btropts_CVEM.ntopics+2,:])
end


mean_TEs = vec(mean(Matrix(TE_Krobustness_df[:,(length(Ks)+2):(2*length(Ks)+1)]), dims = 1))
plot(Ks, mean_TEs, title = "Booking semi-synth, with CVEM")


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
    plt_title = "Booking LDA", fontsize = 10, nwords = 10, title_size = 10)
if save_files; savefig("figures/Booking_BTR/Booking_LDA.pdf"); end;


TE_post_df[:,"LDA"] = sort(ldamodel.ω_post[ldaopts.ntopics+2,:])

## Repeat for many K
for K in Ks
    display("Estimating with "*string(K)*" topics")
    ldaopts = deepcopy(opts)
    ldaopts.ntopics = K
    ldaopts.mse_conv = 2

    ldacrps_tr = create_btrcrps(all_data, ldaopts.ntopics)
    ldamodel = BTRModel(crps = ldacrps_tr, options = ldaopts)
    ## Estimate BTR with EM-Gibbs algorithm
    ldamodel = LDAGibbs(ldamodel)

    TE_Krobustness_df[:,"LDA_K"*string(K)] = sort(ldamodel.ω_post[ldaopts.ntopics+2,:])
end






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
    plt_title = "Booking sLDA + BLR", fontsize = 10, nwords = 10, title_size = 10)
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
