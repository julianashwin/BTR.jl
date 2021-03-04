"""
Implementation of BTR with Gibbs sampling for Yelp dataset
"""

cd("/Users/julianashwin/Documents/GitHub/BTR.jl/")

"""
To make sure the latest version of the package is used run
pkg> dev /Users/julianashwin/Documents/GitHub/BTR.jl
or
pkg> dev https://github.com/julianashwin/BTR.jl

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
display(plot(df.date[1:200], df.stars[1:200], label = ["stars"], legend = :bottomleft,xguidefontsize=8))

## Toggle whether to save the various figures output throughout
save_files = false

"""
Generate a sentiment score from unstemmed documents
"""
## Create a sentiment score for each review using the Harvard Inqiurer lists
df.text = string.(df.text)
df.sentiment = sentimentscore(df.text, HIV_dicts)
ols = lm(@formula(stars ~ sentiment + stars_av_u+ stars_av_b), df)
display(ols)


"""
Create document ids from either review id or business id
"""
## The ids should be in Int64 format
df.review_id = string.(df.review_id)
df[!,:doc_idx] = 1:nrow(df)
#df[!,:doc_idx] = convert_to_ids(df.business_id)
#sort!(df, [:doc_idx])
#showtable(df)




"""
Prepare data for estimation
"""
## Create labels and covariates
x = group_mean(Array{Float64,2}(hcat(df.sentiment,df.stars_av_u,df.stars_av_b)),df.doc_idx)
y = group_mean(Array{Float64,1}(df.stars), df.doc_idx)
docidx_vars = df.doc_idx
docidx_dtm = df.doc_idx
D = length(unique(docidx_dtm))

## Use TextAnalysis package to create a DTM
docs = StringDocument.(df.text_clean)
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
if save_files; savefig("figures/Yelp_BTR/Yelp_trainsplit.pdf"); end;


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
btropts.ntopics = 20
## LDA priors
btropts.α=0.5
btropts.η=0.01
## BLR priors
btropts.μ_ω = 0. # coefficient mean
btropts.σ_ω = 2. # coefficient variance
btropts.a_0 = 4. # residual shape: higher moves mean closer to zero
btropts.b_0 = 2. # residual scale: higher is more spread out
# Plot the prior distribution for residual variance (in case unfamiliar with InverseGamma distributions)
# mean will be b_0/(a_0 - 1)
plot(InverseGamma(btropts.a_0, btropts.b_0), xlim = (0,2), title = "Residual variance prior",
    label = "Prior on residual variance")
scatter!([var(train_data.y)],[0.],label = "Unconditional variance")
if save_files; savefig("figures/Yelp_BTR/Yelp_IGprior.pdf"); end;

## Number of iterations and convergence tolerance
btropts.E_iters = 100 # E-step iterations (sampling topic assignments, z)
btropts.M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
btropts.EM_iters = 25 # Maximum possible EM iterations (will stop here if no convergence)
btropts.CVEM = :obs # Split for separate E and M step batches (if batch = true)
btropts.CVEM_split = 0.5 # Split for separate E and M step batches (if batch = true)
btropts.burnin = 10 # Burnin for Gibbs samplers

btropts.mse_conv = 1 # Number of previous periods to average over for mse convergence
btropts.ω_tol = 0.015 # Convergence tolerance for regression coefficients ω
btropts.rel_tol = true # Whether to use a relative convergence criteria rather than just absolute
## x variables
btropts.xregs = [1,2,3]
btropts.interactions = Array{Int64,1}([])

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
LR + sLDA multiple runs
"""
regressors_train = hcat(ones(size(train_data.x,1)),train_data.x)
regressors_test = hcat(ones(size(test_data.x,1)),test_data.x)
## Bayesian linear regression
blr_coeffs_post, σ2_post = BLR_Gibbs(train_data.y, regressors_train, iteration = btropts.M_iters,
    m_0 = btropts.μ_ω, σ_ω = btropts.σ_ω, a_0 = btropts.a_0, b_0 = btropts.b_0)
blr_coeffs = Array{Float64,1}(vec(mean(blr_coeffs_post, dims = 2)))

## Add residuals to RawData structs
train_resid = train_data.y - regressors_train*blr_coeffs
test_resid = test_data.y - regressors_test*blr_coeffs
train_data_slda = deepcopy(train_data)
test_data_slda = deepcopy(test_data)
train_data_slda.y = train_resid
test_data_slda.y = test_resid
## Options
sldaopts = deepcopy(btropts)
sldaopts.xregs = Array{Int64}([])
sldaopts.interactions = Array{Int64}([])



nruns = 1
for kk in [10,20,30,50]
    print(join(["\n\n\n",string(kk)," topics\n\n\n"]))
    sldaopts.ntopics = kk

    # a,b = (0,0)
    subdirectory = join(["/Users/julianashwin/Desktop/BTR_runs/Yelp/K",
            string(kk),"/ab0_0/sLDA/run_"])
    sldaopts.a_0 = 0.
    sldaopts.b_0 = 0.
    BTR_multipleruns(train_data_slda, test_data_slda, sldaopts, nruns, subdirectory)

    subdirectory = join(["/Users/julianashwin/Desktop/BTR_runs/Yelp/K",
            string(kk),"/ab3_4/sLDA/run_"])
    sldaopts.a_0 = 3.
    sldaopts.b_0 = 4.
    BTR_multipleruns(train_data_slda, test_data_slda, sldaopts, nruns, subdirectory)

    subdirectory = join(["/Users/julianashwin/Desktop/BTR_runs/Yelp/K",
            string(kk),"/ab1p5_4/sLDA/run_"])
    sldaopts.a_0 = 1.5
    sldaopts.b_0 = 4.
    BTR_multipleruns(train_data_slda, test_data_slda, sldaopts, nruns, subdirectory)

end




nruns = 4
for kk in [100]
    print(join(["\n\n\n",string(kk)," topics\n\n\n"]))
    sldaopts.ntopics = kk
    ## Set subdirectory and number of times you want to run
    subdirectory = join(["/Users/julianashwin/Desktop/BTR_runs/Yelp/K",
        string(kk),"/sLDA/run_"])
    ## Run multiple times (for different hyperparameters change btropts)
    BTR_multipleruns(train_data_slda, test_data_slda, sldaopts, nruns, subdirectory)
end



"""
End of script
"""
