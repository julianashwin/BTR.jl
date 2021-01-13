"""
This file defines the structures that are used in estimating Bayesian Topic Regressions
"""

"""
Document and topic structures used to store the data
"""
module DocStructs

using SparseArrays

# Structure including all the raw data to estimate BTR
mutable struct BTRRawData
    dtm::SparseMatrixCSC{Int64,Int64}
    docidx_dtm::Array{Int64,1}
    docidx_vars::Array{Int64,1}
    y::Array{Float64,1}
    x::Array{Float64,2}
    N::Int64
    V::Int64
    vocab::Array{String,1}
end
BTRRawData(dtm, docidx_dtm, docidx_vars, y, x::Array{Float64,2}, vocab::Array{String,1}) = BTRRawData(dtm, docidx_dtm, docidx_vars, y, x, length(y), size(dtm,2), vocab::Array{String,1})
BTRRawData(dtm, docidx_dtm, docidx_vars, y, x, N::Int64, vocab::Array{String,1}) = BTRRawData(dtm, docidx_dtm, docidx_vars, y, x, N, size(dtm,2), vocab::Array{String,1})
BTRRawData(dtm, docidx_dtm, docidx_vars, y, N::Int64, vocab::Array{String,1}) = BTRRawData(dtm, docidx_dtm, docidx_vars, y, zeros(1,1), N, size(dtm,2), vocab::Array{String,1})

mutable struct BTCRawData
    dtm::SparseMatrixCSC{Int64,Int64}
    docidx_dtm::Array{Int64,1}
    docidx_vars::Array{Int64,1}
    y::Array{Int64,1}
    x::Array{Float64,2}
    N::Int64
    V::Int64
    vocab::Array{String,1}
end
BTCRawData(dtm, docidx_dtm, docidx_vars, y, x::Array{Float64,2}, vocab::Array{String,1}) = BTCRawData(dtm, docidx_dtm, docidx_vars, y, x, length(y), size(dtm,2), vocab::Array{String,1})
BTCRawData(dtm, docidx_dtm, docidx_vars, y, x, N::Int64, vocab::Array{String,1}) = BTCRawData(dtm, docidx_dtm, docidx_vars, y, x, N, size(dtm,2), vocab::Array{String,1})
BTCRawData(dtm, docidx_dtm, docidx_vars, y, N::Int64, vocab::Array{String,1}) = BTCRawData(dtm, docidx_dtm, docidx_vars, y, zeros(1,1), N, size(dtm,2), vocab::Array{String,1})



# Structure to keep track of assignments at the paragraph level
mutable struct TopicBasedDocument
    topic::Vector{Int}
    text::Vector{Int}
    topicidcount::Vector{Int}
end
TopicBasedDocument(ntopics) = TopicBasedDocument(Vector{Int}(), Vector{Int}(), zeros(Int, ntopics))

# Structure to keep track of assignments at the document level
mutable struct BTRParagraphDocument
    paragraphs::Vector{TopicBasedDocument}
    topicidcount::Vector{Int}
    y::Float64
    x::Array{Float64,2}
    idx::Int64
end
BTRParagraphDocument(ntopics::Int64, y::Float64, x::Array{Float64,2}, P::Int64, docidx::Int64) = BTRParagraphDocument(
    Vector{TopicBasedDocument}(undef,P), zeros(Int, ntopics), y, x, docidx)

mutable struct BTCParagraphDocument
        paragraphs::Vector{TopicBasedDocument}
        topicidcount::Vector{Int}
        y::Int64
        x::Array{Float64,2}
        idx::Int64
    end
BTCParagraphDocument(ntopics::Int64, y::Int64, x::Array{Float64,2}, P::Int64, docidx::Int64) = BTCParagraphDocument(
        Vector{TopicBasedDocument}(undef,P), zeros(Int, ntopics), y, x, docidx)



# Structure to keep track of assignments at the corpus level
mutable struct Topic
    count::Int
    wordcount::Dict{Int, Int}
end
Topic() = Topic(0, Dict{Int, Int}())


mutable struct BTRCorpus
    docs::Vector{BTRParagraphDocument}
    topics::Vector{Topic}
    docidx_labels::Vector{Int64}
    N::Int64
    ntopics::Int64
    V::Int64
    vocab::Array{String,1}
end
BTRCorpus() = BTRCorpus(Vector{BTRParagraphDocument}(undef,0),
Vector{Topic}(undef,0), Vector{Int64}([]), 0, 0, 0, [""])

mutable struct BTCCorpus
    docs::Vector{BTCParagraphDocument}
    topics::Vector{Topic}
    docidx_labels::Vector{Int64}
    N::Int64
    ntopics::Int64
    V::Int64
    C::Int64
    vocab::Array{String,1}
end
BTCCorpus() = BTCCorpus(Vector{BTCParagraphDocument}(undef,0),
Vector{Topic}(undef,0), Vector{Int64}([]), 0, 0, 0, 2, [""])




end



"""
Structures for the actual estimation
"""

# Estimation options for Bayesian Topic Regression
@with_kw mutable struct BTROptions
    # Model options
    ntopics::Int64 = 2;
    interactions::Array{Int64,1} = Array{Int64,1}([]);
    xregs::Array{Int64,1} = Array{Int64,1}([]);
    emptydocs::Symbol = :prior; # can also be set to :zero
    interlevel::Symbol = :doc; # can also be set to :para
    # Estimation options
    E_iters::Int64 = 100;
    M_iters::Int64 = 250;
    EM_iters::Int64 = 10;
    fullGibbs_iters::Int64 = 10000;
    fullGibbs_thinning::Int64 = 10;
    burnin::Int64 = 10;
    ω_tol::Float64 = 0.01;
    rel_tol::Bool = false;
    CVEM::Symbol = :none; # can also be set to :obs or :paras
    CVEM_split::Float64 = 0.75;
    # Priors
    α::Float64 = 1.;
    η::Float64 = 1.;
    σ_ω::Float64 = 1.;
    μ_ω::Float64 = 0.;
    a_0::Float64 = 0.;
    b_0::Float64 = 0.;
    # Output options
    plot_ω = Bool(true) # = false
end

# Estimation options for Bayesian Topic Classification
@with_kw mutable struct BTCOptions
    # Model options
    ntopics::Int64 = 2;
    interactions::Array{Int64,1} = Array{Int64,1}([]);
    xregs::Array{Int64,1} = Array{Int64,1}([]);
    emptydocs::Symbol = :prior; # can also be set to :zero
    interlevel::Symbol = :doc; # can also be set to :para
    # Estimation options
    E_iters::Int64 = 100;
    M_iters::Int64 = 250;
    EM_iters::Int64 = 10;
    fullGibbs_iters::Int64 = 10000;
    fullGibbs_thinning::Int64 = 10;
    burnin::Int64 = 10;
    ω_tol::Float64 = 0.01;
    rel_tol::Bool = false;
    CVEM::Symbol = :none; # can also be set to :obs or :paras
    CVEM_split::Float64 = 0.75;
    # Priors
    α::Float64 = 1.;
    η::Float64 = 1.;
    # Output options
    plot_ω = Bool(true) # = false
end

# Bayesian Topic Regression model
@with_kw mutable struct BTRModel
    options::BTROptions = BTROptions();
    crps::DocStructs.BTRCorpus = DocStructs.BTRCorpus();
    β::Array{Float64,2} = zeros(options.ntopics,crps.V);
    Z_bar::Array{Float64,2} = zeros(options.ntopics,crps.N);
    ω::Array{Float64,1} = options.μ_ω.*ones(options.ntopics+length(options.interactions)*options.ntopics
     + (length(options.xregs) - length(options.interactions)));
    σ2::Float64 = 1.0;
    regressors::Array{Float64,2} = zeros(crps.N,length(ω));
    y::Array{Float64,1} = vcat(getfield.(crps.docs, :y)...);
    ω_post::Array{Float64,2} = options.μ_ω.+ sqrt(options.σ_ω)*randn(length(ω),options.M_iters);
    σ2_post::Array{Float64,1} = zeros(options.E_iters);
    ω_iters::Array{Float64,2} = zeros(length(ω), options.EM_iters+1);
    pplxy::Float64 = 0.
end

# Bayesian Topic Classification model
@with_kw mutable struct BTCModel
    options::BTCOptions = BTCOptions();
    crps::DocStructs.BTCCorpus = DocStructs.BTCCorpus();
    β::Array{Float64,2} = zeros(options.ntopics,crps.V);
    Z_bar::Array{Float64,2} = zeros(options.ntopics,crps.N);
    ω::Array{Float64,1} = zeros(options.ntopics+length(options.interactions)*options.ntopics
     + (length(options.xregs) - length(options.interactions)));
    Σ::Array{Float64,2} = Array{Float64,2}(I(length(ω)));
    regressors::Array{Float64,2} = zeros(crps.N,length(ω));
    y::Array{Int64,1} = vcat(getfield.(crps.docs, :y)...);
    ω_iters::Array{Float64,2} = zeros(length(ω), options.EM_iters+1);
    dev::Float64 = 0.;
    pplxy::Float64 = 0.
end

# Bayesian Topic Regression predictions
@with_kw mutable struct BTRPrediction
    options::BTROptions = BTROptions();
    crps::DocStructs.BTRCorpus = DocStructs.BTRCorpus();
    Z_bar::Array{Float64,2} = zeros(options.ntopics,crps.N);
    regressors::Array{Float64,2} = zeros(crps.N,(options.ntopics +
        options.ntopics*length(options.interactions) + length(options.xregs) - length(options.interactions)));
    y_pred::Array{Float64,1} = zeros(crps.N);
    y::Array{Float64,1} = vcat(getfield.(crps.docs, :y)...);
    pplxy::Float64 = 0.
end

# Bayesian Topic Classification predictions
@with_kw mutable struct BTCPrediction
    options::BTCOptions = BTCOptions();
    crps::DocStructs.BTCCorpus = DocStructs.BTCCorpus();
    Z_bar::Array{Float64,2} = zeros(options.ntopics,crps.N);
    regressors::Array{Float64,2} = zeros(crps.N,(options.ntopics +
        options.ntopics*length(options.interactions) + length(options.xregs) - length(options.interactions)));
    p_pred::Array{Float64,1} = zeros(crps.N);
    y_pred::Array{Int64,1} = zeros(crps.N);
    y::Array{Int64,1} = vcat(getfield.(crps.docs, :y)...);
    pplxy::Float64 = 0.
end



"""
End of script
"""
