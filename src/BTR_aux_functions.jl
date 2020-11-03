"""
This file defines some auxilliary functions used throughout the package
"""

using Plots, Plots.PlotMeasures

"""
Efficient multinomial sampler that returns index of chosen bin
"""
function multinomial_draw(p::Array{Float64,1})
    threshold = rand()::Float64
    sum_prob = 0.0::Float64
    new_id = 0::Int64
    for (topicid, prob) in enumerate(p)
        sum_prob += prob
        if sum_prob > threshold
            new_id = topicid::Int64
            break
        end
    end
    return new_id
end

"""
Replace all NaN with float
"""
function replace_nan!(x::Array{Float64}, replacement::Float64)
    for i = eachindex(x)
        if isnan(x[i])
            x[i] = replacement
        end
    end
end

"""
Function that creates interaction effects
    Order is Z.*x[1], Z.*x[2], Z.*x[3] etc...
"""
function create_inter_effects(Z::Array{Float64,2},x::Array{Float64,2},ntopics::Int)

    inter_effects::Array{Float64,2} = zeros(size(x,1),ntopics*size(x,2))
    for jj in 1:size(x,2)
        col_range= (jj*ntopics-ntopics+1):(jj*ntopics)
        inter_effects[:,col_range] = transpose(Z).*vec(x[:,jj])
    end
    return inter_effects
end

"""
Function that computes group level means for 2d array
"""
function group_mean(x::Array{Float64,2},groups::Array{Int64,1})
    @assert size(x,1) == length(groups) "Variable and Group vectors must be same length"


    group_order = unique(groups)
    x_new = zeros(length(group_order),size(x,2))
    ii=1
    for gg in group_order
        idxs = (groups.==gg)
        x_new[ii,:] .= vec(mean(x[idxs,:],dims=1))
        ii+=1
    end

    return x_new
end

"""
Function that computes group level sum for 2d array
"""
function group_sum(x::Array{Float64,2},groups::Array{Int64,1})
    @assert size(x,1) == length(groups) "Variable and Group vectors must be same length"


    group_order = unique(groups)
    x_new = zeros(length(group_order),size(x,2))
    ii=1
    for gg in group_order
        idxs = (groups.==gg)
        x_new[ii,:] .= vec(sum(x[idxs,:],dims=1))
        ii+=1
    end

    return x_new
end


"""
Function that finds first element of each group in ordered array of group membership
"""
function findfirst_group(groups)
    Ns = counts(groups)
    first_group = zeros(length(unique(groups)))
    ii = 1
    for gg in 1:length(first_group)
        first_group[gg] = ii
        ii += Ns[gg]
    end
    return Int.(first_group)
end


"""
Function that generate bar topics for synthetic data
"""
function bartopics(η::Float64, K::Int64, V::Int64)
    β::Array{Float64,2} = η.*zeros(K,V)
    toplength::Int64 = floor(Int, V/K)
    start::Int64 = 1
    for kk in 1:K
        if kk < K
            β[kk,start:(start+toplength-1)] .+= 50
            start += toplength
        elseif kk == K
            β[kk,start:V] .+= 50
        end
    end
    β ./= sum(β, dims=2)
    return β
end


"""
Function that generates string documents and topic assignments
"""
function generate_docs(D, Nd, K, θ_true, β_true)
    docs = Array{String,1}(undef, D)
    Z = Int.(zeros(D,Nd))
    V = size(β_true, 2)
    topic_counts = Int.(zeros(D,K))
    word_counts = Int.(zeros(D,V))
    for dd in 1:D
        doc = ""
        for nn in 1:Nd
            topic = multinomial_draw(θ_true[:,dd])
            Z[dd,nn] = topic
            topic_counts[dd,topic] +=1
            word = multinomial_draw(β_true[topic,:])
            word_counts[dd, word] += 1
            doc = join([doc, string(word)], " ")
        end
        docs[dd] = doc
    end
    return docs, Z, topic_counts, word_counts
end



"""
Closed form Bayesian Linear Regression with known σ:
BLR(y::Array{Float64,1},x::Array{Float64,2},
        μ::Array{Float64,1},S::Array{Float64,2},σ2::Float64)
The key parameters are:
    y is a vector of the target (dependent) variable
    x is a matrix of the regressor (independent) variables
    μ is the mean of the coefficient prior
    S is the covariance matrix of the coefficient prior
    σ2 is the (known) residual variance
"""
function BLR(y::Array{Float64,1},x::Array{Float64,2},
        μ::Array{Float64,1},S::Array{Float64,2},σ2::Float64)
    Σ_inv = σ2^(-1)*(transpose(x)*x) .+ inv(S)
    m = σ2^(-1)*inv(Σ_inv)*(transpose(x)*y)
    Σ = inv(Σ_inv)
    return m,Σ
end


"""
A Gibbs sampled Bayesian Linear Regression with NIG priors
BLR_Gibbs(y::Array{Float64,1},x::Array{Float64,2};
    m_0::Float64 = 0. ,σ_ω::Float64 = 5.,
    a_0::Float64 = 1., b_0::Float64 = 1.,
    iteration::Int64 = 1000, burnin::Int64 = 20)
Key parameters are:
    y is a vector of the target (dependent) variable
    x is a matrix of the regressor (independent) variables
    m_0 is the mean of the coefficient prior
    σ_ω is the variance of the coefficient prior
    a_0 is the shape parameter of the variance prior
    b_0 is the scale parameter of the variance prior
    fulldist toggles whether to sample from the joint posterior, or just report the expectations
"""

function BLR_Gibbs(y::Array{Float64,1},regressors::Array{Float64,2};
    fulldist = true,
    m_0::Float64 = 0. ,σ_ω::Float64 = 5.,
    a_0::Float64 = 0., b_0::Float64 = 0.,
    iteration::Int64 = 1000, burnin::Int64 = 20)

    # Make sure the data is the right size
    N = size(regressors, 1)::Int64
    J = size(regressors,2)::Int64
    @assert (N == size(y,1)) "Dimensions of target variable and regressors must match"

    # Convert coefficient priors to right format
    m_0_vec = m_0.*ones(J)::Array{Float64,1}
    M_0 = Array{Float64,2}(σ_ω*I(J))::Array{Float64,2}

    # The elements of the posterior can be calculated in advance
    M_n = inv(inv(M_0) +transpose(regressors)*regressors)
    m_n = M_n*(inv(M_0)*m_0_vec + transpose(regressors)*y)
    a_n = a_0 + N/2
    b_n = b_0 + 0.5*(transpose(y)*y + transpose(m_0_vec)*inv(M_0)*m_0_vec -
        transpose(m_n)*inv(M_n)*m_n)


    # We can work out the expectation of β and σ2 analytically
    β = m_n::Array{Float64,1}
    if a_0 <= 1. || b_0 <= 0.
        σ2 = mean((y - regressors*m_n).^2)
    else
        σ2 = b_n/(a_n - 1)::Float64
    end

    # Placeholders for sampled β and σ2
    β_post = repeat(β,1,(iteration+burnin))::Array{Float64,2}
    σ2_post = repeat([σ2],(iteration+burnin))::Array{Float64,1}
    ## Sample from the posterior for β and σ2 if option is set to true
    if fulldist
        if a_0 <= 1. || b_0 <= 0.
            β_dist = MvNormal(β, Matrix(Hermitian(σ2*M_n)))
            β_post = rand(β_dist, iteration+burnin)
        else
            # Marginal distribution of σ2 doesn't depend on β, so we can draw all in one go
            σ2_dist = InverseGamma(a_n, b_n)
            σ2_post = rand(σ2_dist,(iteration + burnin))
            prog = Progress((iteration+burnin - 1), 1)
            for it in 2:(iteration+burnin)
                # Use last iterations σ2
                σ2_it = σ2_post[it-1]
                # Sample β
                β_dist = MvNormal(m_n, Matrix(Hermitian(σ2_it*M_n)))
                β_post[:,it] = rand(β_dist)
                next!(prog)
            end
        end
        β_post = β_post[:,(burnin+1):(burnin+iteration)]
        σ2_post = σ2_post[(burnin+1):(burnin+iteration)]
    end


    return β, σ2, β_post, σ2_post

end



"""
End of script
"""
