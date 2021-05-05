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
    nwords = vec(sum(dtm_sparse.dtm, dims = 2))
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
    df.synth_y = 1.0.*df.av_score - 0.5*df.Leisure +
        10.0.*df.pos_prop + 0.1.*randn(nrow(df))

    display(cor(hcat(df.Leisure, df.pos_prop, df.sentiment, nwords)))

    ## Export
    CSV.write("data/booking_semisynth_sample_v2.csv", df)
end

display(cor(hcat(df.Leisure, df.av_score, df.pos_prop, df.synth_y)))





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
opts.CVEM_split = 0.7 # Split for separate E and M step batches (if batch = true)
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
Dataframes to fill
"""
## Settings
nruns = 20
opts.M_iters = 2500
Ks = [2,5,10,15,20,25,30,40,50]

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
## Topics
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

    CSV.write("data/semisynth_booking/TE_MBTR_post.csv", TE_Krobustness_df)
    CSV.write("data/semisynth_booking/TE_MBTR_median.csv", TE_Krobustness_medians_df)
end

"""
Estimate BTR without CVEM
"""
## Repeat for many K
TE_Krobustness_df = DataFrame(NoText_reg = sort(blr_notext_coeffs_post[3,:]))
Ks = [2,5,10,15,20,25,30,40,50]
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

    ## Bayesian linear regression on training set
    blr_ω, blr_σ2, blr_ω_post, blr_σ2_post = BLR_Gibbs(ldamodel.y, ldamodel.regressors,
        m_0 = ldaopts.μ_ω, σ_ω = ldaopts.σ_ω, a_0 = ldaopts.a_0, b_0 = ldaopts.b_0,
        iteration = ldaopts.M_iters)
    ldamodel.ω = blr_ω
    ldamodel.ω_post = blr_ω_post


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
    slda2opts.CVEM = :none
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
if save_files
    #CSV.write("data/semisynth_booking/TE_Krobustness.csv",TE_Krobustness_df)
end

#TE_Krobustness_df = CSV.read("data/semisynth_booking/TE_Krobustness.csv",DataFrame)

scholar_CVEM_df = CSV.read("data/semisynth_booking/scholar/semisynth_booking_scholar_regweight_bootstrap_cv5050.csv", DataFrame)
sort!(scholar_CVEM_df, :Column1)


scholar_df = CSV.read("data/semisynth_booking/scholar/semisynth_booking_scholar_regweight_bootstrap_noCV.csv", DataFrame)
sort!(scholar_df, :Column1)


"""
Plot treatment effects
"""
## Identify columns for each model
NoText_cols = occursin.("NoText",names(TE_Krobustness_df))
BTR_cols = occursin.("BTR_noCVEM",names(TE_Krobustness_df))
BTR_CVEM_cols = occursin.("BTR_CVEM",names(TE_Krobustness_df))
LDA_cols = occursin.("LDA_",names(TE_Krobustness_df)) .& .!(occursin.("s",names(TE_Krobustness_df)))
sLDA_cols = occursin.("sLDA_",names(TE_Krobustness_df))

## Function to plot estimate with ccredible intervals
est_df = TE_Krobustness_df
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

    plot!(Ks, med_ests, color = est_color, label = label)
    scatter!(Ks, med_ests, color = est_color, label = "")
    plot!(Ks, med_ests, ribbon=(upper_ests.- med_ests, med_ests.- lower_ests),
        color = est_color, label = "", fillalpha = 0.5)

end


model_names = ["BTR (no CVEM)","BTR (CVEM)", "LDA", "sLDA", "NoText BLR"]
nmodels = length(model_names)
plt1 = plot(legend = false, xlim = (0,maximum(Ks)+2), ylim = (-1.0, 0.5),
    xlabel = "Number of Topics", ylabel = "Estimate Treatment Effect",
    title = "Booking semi-synth")
plot!([0.,(Float64(maximum(Ks))+2.0)],[-0.5,-0.5], linestyle = :dash,color =:red,
    label = "Ground truth", legend = :topright)
# Add various model estimates
plot_estimates(TE_Krobustness_df, Ks, "No Text LR", NoText_cols, :grey)
plot_estimates(TE_Krobustness_df, Ks, "MBTR", BTR_cols, :blue)
plot_estimates(TE_Krobustness_df, Ks, "MBTR (CVEM)", BTR_CVEM_cols, :lightblue)
plot_estimates(TE_Krobustness_df, Ks, "LDA", LDA_cols, :green)
plot_estimates(TE_Krobustness_df, Ks, "sLDA", sLDA_cols, :orange)
# Add scholar results
plot!([2,5,10,15,20,25,30,40,50], scholar_df.w1_median, color = :pink, label = "SCHOLAR")
scatter!([2,5,10,15,20,25,30,40,50], scholar_df.w1_median, color = :pink, label = "")
plot!([2,5,10,15,20,25,30,40,50], scholar_df.w1_median, ribbon=(scholar_df.w1_upper.-
    scholar_df.w1_median, scholar_df.w1_median.- scholar_df.w1_lower),
    color = :pink, label = "", fillalpha = 0.5)

plot!([2,5,10,15,20,25,30,40,50], scholar_CVEM_df.w1_median, color = :purple, label = "SCHOLAR (CV)")
scatter!([2,5,10,15,20,25,30,40,50], scholar_CVEM_df.w1_median, color = :purple, label = "")
plot!([2,5,10,15,20,25,30,40,50], scholar_CVEM_df.w1_median, ribbon=(scholar_CVEM_df.w1_upper.-
    scholar_CVEM_df.w1_median, scholar_CVEM_df.w1_median.- scholar_CVEM_df.w1_lower),
    color = :purple, label = "", fillalpha = 0.5)

plot!(size = (500,400))


savefig("figures/semisynth/Booking_TEs.pdf")







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
