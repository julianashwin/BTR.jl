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



## Toggle whether to save the various figures output throughout
save_files = false
regenerate_data = false
γ_1 = 0.5
## Load data
df = CSV.read("data/yelp_semisynth_sample_gamma"*string(γ_1)*".csv", DataFrame, threaded = false)
## Check correlations
display(cor(hcat(df.synth_y, df.stars_av_b, df.US, df.harvard_score)))

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
Create synthetic variables
"""

if regenerate_data

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


    ## Set γ_1 for strength of correlation with confounder
    γ_1 = 0.0
    γ_0 = 0.0
    σ_y_true = 0.1

    df[:,"US"] .= 0.
    df[:,"PrUS"] .= exp.(γ_0 .+ γ_1.*df.harvard_score)./(1.0 .+ exp.(γ_0 .+ γ_1.*df.harvard_score))

    df[:,"US"] .= Float64.(rand.(Binomial.(1,df.PrUS)))

    mean(df.US)
    df[:,"synth_y"] = 1.0.*df.stars_av_b + 1.0.*df.harvard_score - 1.0 .* df.US + 0.1.*randn(nrow(df))

    display(cor(hcat(df.synth_y, df.stars_av_b, df.US, df.harvard_score)))


    ## Export
    #CSV.write("data/yelp_semisynth_sample_gamma"*string(γ_1)*".csv", df)
end


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

display(cor(hcat(df.synth_y, df.US, df.harvard_score)))

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
opts.ntopics = 30
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
opts.CVEM_split = 0.7 # Split for separate E and M step batches (if batch = true)

## Comvergence options
opts.mse_conv = 2
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


"""
Dataframes to fill
"""
## Settings
nruns = 10
opts.M_iters = 2500
Ks = [5,10,20,30,50]

## No text regression as baseline benchmark
blr_coeffs, blr_σ2, blr_notext_coeffs_post, σ2_post = BLR_Gibbs(all_data.y, regressors_notext,
    iteration = opts.M_iters, m_0 = opts.μ_ω, σ_ω = opts.σ_ω, a_0 = opts.a_0, b_0 = opts.b_0)

## Dataframe with all samples across multiple runs
TE_Krobustness_df = DataFrame(NoText_reg = sort(repeat(blr_notext_coeffs_post[3,:], nruns)))
for k in Ks
    TE_Krobustness_df[:,Symbol("BTR_noCVEM_K"*string(k))] .= 0.
end
## Dataframe with just median across multiple runs
TE_Krobustness_medians_df = DataFrame(run = 1:nruns)
for k in Ks
    TE_Krobustness_medians_df[:,Symbol("BTR_noCVEM_K"*string(k))] .= 0.
end




"""
Estimate BTR without CVEM
"""
# Multiple runs for each K
for nn in 1:nruns
    for K in Ks
        display("Estimating BTR without CVEM with "*string(K)*" topics for the "*string(nn)*"th time")
        btropts_noCVEM = deepcopy(opts)
        btropts_noCVEM.CVEM = :none
        btropts_noCVEM.ntopics = K
        btropts_noCVEM.mse_conv = 2

        btrcrps_tr = create_btrcrps(all_data, btropts_noCVEM.ntopics)
        btrmodel_noCVEM = BTRModel(crps = btrcrps_tr, options = btropts_noCVEM)

        ## Estimate BTR with EM-Gibbs algorithm
        btrmodel_noCVEM = BTRemGibbs(btrmodel_noCVEM)

        ## Save posterior dist of treatment effect
        obs = (1+((nn-1)*opts.M_iters)):(nn*opts.M_iters)
        TE_Krobustness_df[obs,"BTR_noCVEM_K"*string(K)] = sort(btrmodel_noCVEM.ω_post[btropts_noCVEM.ntopics+2,:])
        # Save median of treatment effect estimate
        TE_Krobustness_medians_df[nn,Symbol("BTR_noCVEM_K"*string(K))] =
            median(btrmodel_noCVEM.ω_post[btropts_noCVEM.ntopics+2,:])
    end

    CSV.write("data/semisynth_yelp/TE_MBTR_post_gamma"*string(γ_1)*".csv",
        TE_Krobustness_df)
    CSV.write("data/semisynth_yelp/TE_MBTR_median_gamma"*string(γ_1)*".csv",
        TE_Krobustness_medians_df)
end

#mean_TEs = vec(mean(Matrix(TE_Krobustness_df[:,(2:(length(Ks)+1))]), dims = 1))
#plot(Ks, mean_TEs, title = "Yelp semi-synth, without CVEM")


"""
Estimate BTR with CVEM
"""
## Repeat for many K
for K in Ks
    display("Estimating BRT with CVEM with "*string(K)*" topics")
    btropts_CVEM = deepcopy(opts)
    btropts_CVEM.ntopics = K
    btropts_CVEM.mse_conv = 2

    btrcrps_tr = create_btrcrps(all_data, btropts_CVEM.ntopics)
    btrmodel_CVEM = BTRModel(crps = btrcrps_tr, options = btropts_CVEM)
    ## Estimate BTR with EM-Gibbs algorithm
    btrmodel_CVEM = BTRemGibbs(btrmodel_CVEM)

    ## Save posterior dist of treatment effect
    TE_Krobustness_df[:,"BTR_CVEM_K"*string(K)] = sort(btrmodel_CVEM.ω_post[btropts_CVEM.ntopics+2,:])
end

#mean_TEs = vec(mean(Matrix(TE_Krobustness_df[:,(length(Ks)+2):(2*length(Ks)+1)]), dims = 1))
#plot(Ks, mean_TEs, title = "Yelp semi-synth, with CVEM")



"""
Estimate 2 stage LDA then Bayesian Linear Regression (BLR)
    In the synthetic data this does about as well as BTR because the
    text is generated from an LDA model.
"""
## Repeat for many K
for K in Ks
    display("Estimating LDA with "*string(K)*" topics")
    ldaopts = deepcopy(opts)
    ldaopts.ntopics = K
    ldaopts.fullGibbs_iters = 1000
    ldaopts.fullGibbs_thinning = 2
    ldaopts.burnin = 50

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

    ## Save posterior dist of treatment effect
    TE_Krobustness_df[:,"LDA_K"*string(K)] = sort(ldamodel.ω_post[ldaopts.ntopics+2,:])
end


"""
Estimate 2 stage slDA then BLR on residuals
"""
## Repeat for many K
for K in Ks
    display("Estimating sLDA with "*string(K)*" topics")
    ## Set options sLDA on residuals
    slda2opts = deepcopy(opts)
    slda2opts.xregs = Array{Int64,1}([])
    slda2opts.interactions = Array{Int64,1}([])

    ## Initialise BTRModel object
    slda2crps_tr = create_btrcrps(all_data, slda2opts.ntopics)
    slda2model = BTRModel(crps = slda2crps_tr, options = slda2opts)

    ## Estimate sLDA on residuals
    slda2model = BTRemGibbs(slda2model)

    ## Identify residuals to train second stage regression
    residuals_slda = train_data.y .- slda2model.regressors*slda2model.ω

    ## Bayesian linear regression on training set
    regressors_slda = hcat(ones(size(train_data.x,1)),train_data.x)
    # No fixed effect, no batch
    blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(residuals_slda, regressors_slda,
        m_0 = slda2opts.μ_ω, σ_ω = slda2opts.σ_ω, a_0 = slda2opts.a_0, b_0 = slda2opts.b_0,
        iteration = slda2opts.M_iters)

    slda2_TE = blr_ω[3]

    TE_Krobustness_df[:,"sLDA_K"*string(K)] = sort(blr_ω_post[3,:])

end


"""
Export estimated treatment effects
"""

#CSV.write("data/semisynth_yelp/TE_Krobustness_gamma"*string(γ_1)*".csv",
#    TE_Krobustness_df)

#TE_Krobustness_df1 = CSV.read("data/semisynth_yelp/TE_Krobustness_gamma"*string(γ_1)*".csv",DataFrame)
#TE_Krobustness_df = CSV.read("data/semisynth_yelp/TE_MBTR_post_gamma"*string(γ_1)*".csv",
#    DataFrame, threaded = false)
for K in Ks
    TE_Krobustness_df[:,"BTR_noCVEM_K"*string(K)] = sort(TE_Krobustness_df[:,"BTR_noCVEM_K"*string(K)])
end


scholar_df = CSV.read("data/semisynth_yelp/scholar/semisynth_yelp_scholar_regweight_bootstrap_noCV_gamma"*
    string(γ_1)*".csv", DataFrame)
sort!(scholar_df, :Column1)


"""
Plot treatment effects
"""
## Identify columns for each model
NoText_cols = occursin.("NoText",names(TE_Krobustness_df1))
BTR_cols = occursin.("BTR_noCVEM",names(TE_Krobustness_df))
#BTR_CVEM_cols = occursin.("BTR_CVEM",names(TE_Krobustness_df))
LDA_cols = occursin.("LDA_",names(TE_Krobustness_df1)) .& .!(occursin.("s",names(TE_Krobustness_df1)))
sLDA_cols = occursin.("sLDA_",names(TE_Krobustness_df1))

## Function to plot estimate with ccredible intervals
est_df = TE_Krobustness_df1
cols = NoText_cols
label = "No Text LR"
est_color = :grey
function plot_estimates(est_df, Ks, label, cols, est_color)
    med_row = Int(nrow(est_df)/2)
    low_row = Int(round(nrow(est_df)*0.025,digits = 0))
    high_row = Int(round(nrow(est_df)*0.975, digits = 0))

    if sum(cols) == 1
        med_ests = repeat(Array(est_df[med_row, cols]), length(Ks))
        upper_ests = repeat(Array(est_df[high_row, cols]), length(Ks))
        lower_ests = repeat(Array(est_df[low_row, cols]), length(Ks))
    else
        med_ests = Array(est_df[med_row, cols])
        upper_ests = Array(est_df[high_row, cols])
        lower_ests = Array(est_df[low_row, cols])
    end

    Plots.plot!(Ks, med_ests, color = est_color, label = label)
    Plots.scatter!(Ks, med_ests, color = est_color, label = "")
    Plots.plot!(Ks, med_ests, ribbon=(med_ests.- lower_ests, upper_ests.- med_ests),
        color = est_color, label = "", fillalpha = 0.5)

end


Ks = [5,10,20,30,50]
pyplot()
model_names = ["BTR (no CVEM)","BTR (CVEM)", "LDA", "sLDA", "NoText BLR"]
nmodels = length(model_names)
plt1 = Plots.plot(legend = false, xlim = (0,maximum(Ks)+2), ylim = (-1.1, 0.0),
    xlabel = "Number of Topics", ylabel = "Estimate Treatment Effect",
    title = "Yelp semi-synth "*raw"$\gamma_1$ = "*string(γ_1))
Plots.plot!([0.,(Float64(maximum(Ks))+2.0)],[-1.,-1.], linestyle = :dash,color =:red,
    label = "Ground truth", legend = :topright)
# Add various model estimates
plot_estimates(TE_Krobustness_df1, Ks, "No Text LR", NoText_cols, :grey)
plot_estimates(TE_Krobustness_df, Ks, "BTR", BTR_cols, :blue)
#plot_estimates(TE_Krobustness_df, Ks, "MBTR (CVEM)", BTR_CVEM_cols, :lightblue)
plot_estimates(TE_Krobustness_df1, Ks, "LDA", LDA_cols, :green)
plot_estimates(TE_Krobustness_df1, Ks, "sLDA", sLDA_cols, :orange)
# Add scholar results
Plots.plot!(Ks, scholar_df.w1_median, color = :pink, label = "rSCHOLAR")
Plots.scatter!(Ks, scholar_df.w1_median, color = :pink, label = "")
Plots.plot!(Ks, scholar_df.w1_median, ribbon=(scholar_df.w1_upper.-
    scholar_df.w1_median, scholar_df.w1_median.- scholar_df.w1_lower),
    color = :pink, label = "", fillalpha = 0.5)

#plot!(Ks, scholar_CVEM_df.w1_median, color = :purple, label = "SCHOLAR (CV)")
#scatter!(Ks, scholar_CVEM_df.w1_median, color = :purple, label = "")
#plot!(Ks, scholar_CVEM_df.w1_median, ribbon=(scholar_CVEM_df.w1_upper.-
#    scholar_CVEM_df.w1_median, scholar_CVEM_df.w1_median.- scholar_CVEM_df.w1_lower),
#    color = :purple, label = "", fillalpha = 0.5)

Plots.plot!(size = (300,400))


Plots.savefig("figures/semisynth/Yelp_gamma"*string(γ_1)*".pdf")







"""
End of script
"""
