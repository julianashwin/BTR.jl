"""
Function that performs the M-step, of estimating ω and σ2, on a BTRModel object
"""
function BTRMstep(btrmodel::BTRModel)
    ## Extract options
    opts::BTROptions = btrmodel.options
    ntopics::Int64 = opts.ntopics
    xregs::Array{Int64,1} = opts.xregs
    interactions::Array{Int64,1} = opts.interactions
    nointeractions::Array{Int64,1} = setdiff(xregs, interactions)

    ## Extract target variables
    y::Array{Float64,1} = vcat(getfield.(btrmodel.crps.docs, :y)...)
    ## Extract regressors
    Z_bar::Array{Float64,2} = Matrix(transpose(btrmodel.Z_bar))
    x = vcat(getfield.(btrmodel.crps.docs, :x)...)
    inter_effects::Array{Float64,2} = zeros(length(y), ntopics*length(interactions))::Array{Float64,2}
    for jj in 1:length(interactions)
        col_range::UnitRange{Int64} = ((jj*ntopics-ntopics+1):(jj*ntopics))
        inter_effects[:,col_range] = Z_bar.*vec(x[:,interactions[jj]])
    end
    regressors::Array{Float64,2} = hcat(Z_bar, inter_effects, x[:,nointeractions])

    # Estimate BLR
    ω, σ2, ω_post, σ2_post = BLR_Gibbs(y, regressors, iteration = opts.M_iters,
        m_0 = opts.μ_ω, σ_ω = opts.σ_ω, a_0 = opts.a_0, b_0 = opts.b_0)

    # repopulate btrmodel with updated parameters
    btrmodel.ω = ω
    btrmodel.σ2 = σ2
    btrmodel.ω_post = ω_post
    btrmodel.σ2_post = σ2_post

    return btrmodel
end
