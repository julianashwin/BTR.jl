# BTR.jl
Estimate Bayesian Topic Regression models

## Importing necessary functions

In order to estimate BTR models and plot their output, you will first need to import the necessary functions. First, make sure you are in the correct working directory (i.e. the one which contains the src folder). Then run the line
```julia
include("src/BTR.jl")
```
which will load both the estimation and visualisation functions into your workspace. The BTR_synthetic.jl file runs through an example in which a synthetic dataset is generated, prepared for estimation and then a variety of models estimated on it. More explanation for this file can be found in the sythetic_example.txt file.

## Pre-processing

Before estimating we need to convert the text-data into a document-term-matrix (DTM) for which we can use the DocumentTermMatrix function from the [link text itself]: http://www.reddit.comTextAnalysis.jl package

The core functions for estimation are
* BTR_EMGibbs
```julia
include("src/BTR.jl")
```
  * Takes as arguments a document-term-matrix encoded as a sparse matrix; the number of topics as an integer; and the response variable as an array of floats.
* BTR_Gibbs_predict
