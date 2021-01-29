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
df = CSV.read("data/booking_posneg_sample.csv", DataFrame, threaded = false)
df = df[.!ismissing.(df.text_clean),:]
# Check that the variables look sensible
display(plot(df.pos[1:200], label = ["UK dummy"], legend = :bottomleft,xguidefontsize=8))

## Toggle whether to save the various figures output throughout
save_files = false

"""
Generate a sentiment score from unstemmed documents
"""
## Create a sentiment score for each review using the Harvard Inqiurer lists
df.text = string.(df.text)
df.sentiment = sentimentscore(df.text, HIV_dicts)


ols = lm(@formula(pos ~ sentiment + Average_Score + Reviewer_Score), df)
display(ols)


"""
Create document ids from either review id or business id
"""
## The ids should be in Int64 format
#df.review_id = string.(df.review_id)
df[!,:doc_idx] = 1:nrow(df)
#df[!,:doc_idx] = convert_to_ids(df.business_id)
#sort!(df, [:doc_idx])
#showtable(df)




"""
Prepare data for estimation
"""
## Create labels and covariates
x = Array{Float64,2}(hcat(df.sentiment,df.Average_Score,df.Reviewer_Score))
y = Array{Int64,1}(df.pos)
docidx_vars = df.doc_idx
docidx_dtm = df.doc_idx
D = length(unique(docidx_dtm))

## Use TextAnalysis package to create a DTM
df.text_clean = string.(df.text_clean)
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
train_data, test_data = btc_traintestsplit(dtm_in, docidx_dtm, docidx_vars, y, vocab, x = x,
    train_split = 0.75, shuffle_obs = false)
# Alternatively, can convert the entier set to BTRRawData with
#all_data = DocStructs.BTRRawData(dtm_in, docidx_dtm, docidx_vars, y, x, vocab)
## Visualise the training-test split
histogram(train_data.docidx_dtm, bins = 1:D, label = "training set",
    xlab = "Observation", ylab= "Paragraphs", c=1, lc=nothing)
histogram!(test_data.docidx_dtm, bins = 1:D, label = "test set", c=2, lc=nothing)
if save_files; savefig("figures/Yelp_BTR/Yelp_trainsplit.pdf"); end;


"""
Standardise using only the training data
"""
## Use mean and std from training data to normalise both sets
x_mean_tr = mean(train_data.x,dims=1)
x_std_tr = std(train_data.x,dims=1)
train_data.x = (train_data.x .- x_mean_tr)./x_std_tr
test_data.x = (test_data.x .- x_mean_tr)./x_std_tr



"""
Set priors and estimation optioncs here to be consistent across models
"""
## Initialiase estimation options
btcopts = BTCOptions()
## Number of topics
btcopts.ntopics = 10
## LDA priors
btcopts.α=0.5
btcopts.η=0.01
## Number of iterations and convergence tolerance
btcopts.E_iters = 100 # E-step iterations (sampling topic assignments, z)
btcopts.M_iters = 2500 # M-step iterations (sampling regression coefficients residual variance)
btcopts.EM_iters = 25 # Maximum possible EM iterations (will stop here if no convergence)
btcopts.CVEM = :obs # Split for separate E and M step batches (if batch = true)
btcopts.CVEM_split = 0.5 # Split for separate E and M step batches (if batch = true)
btcopts.burnin = 10 # Burnin for Gibbs samplers
btcopts.ω_tol = 0.015 # Convergence tolerance for regression coefficients ω
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
sLDA-C multiple runs
"""
## Options
sldaopts = deepcopy(btcopts)
sldaopts.xregs = Array{Int64}([])
sldaopts.interactions = Array{Int64}([])
nruns = 20
for kk in [5,10,20,30,50]
    print(join(["\n\n\n",string(kk)," topics\n\n\n"]))
    sldaopts.ntopics = kk
    ## Set subdirectory and number of times you want to run
    subdirectory = join(["/Users/julianashwin/Desktop/BTR_runs/Booking_posneg/K",
        string(kk),"/sLDA/run_"])
    ## Run multiple times (for different hyperparameters change btropts)
    BTC_multipleruns(train_data, test_data, sldaopts, nruns, subdirectory)
end




"""
End of script
"""
