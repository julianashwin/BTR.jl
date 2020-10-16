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
function bartopics(η::Float64, K::Int64, Y::Int64)
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
    m_0::Float64 = 0. ,V_0::Float64 = 5.,
    a_0::Float64 = 1., b_0::Float64 = 1.,
    iteration::Int64 = 1000, burnin::Int64 = 20)
Key parameters are:
    y is a vector of the target (dependent) variable
    x is a matrix of the regressor (independent) variables
    m_0 is the mean of the coefficient prior
    V_0 is the covariance matrix of the coefficient prior
    a_0 is the shape parameter of the variance prior
    b_0 is the scale parameter of the variance prior
"""
function BLR_Gibbs(y::Array{Float64,1},x::Array{Float64,2};
    m_0::Float64 = 0. ,V_0::Float64 = 5.,
    a_0::Float64 = 1., b_0::Float64 = 1.,
    iteration::Int64 = 1000, burnin::Int64 = 20)

    N::Int64 = size(x, 1)
    J::Int64 = size(x,2)
    @assert (N == size(y,1)) "Dimensions of y and x must match"

    # Define m_0_vec and V_0_inv
    m_0_vec::Array{Float64,1} = m_0.*ones(J)
    V_0_inv::Array{Float64,2} = inv(Array{Float64,2}(V_0*I(J)))

    β::Array{Float64,2} = zeros(J,iteration+burnin)
    σ2::Array{Float64,1} = zeros(iteration+burnin)

    V_n::Array{Float64,2} = inv(transpose(x)*x .+ V_0_inv)
    m_n::Array{Float64,1} = V_n*(V_0_inv*m_0_vec + transpose(x)*y)

    # Initialise chain
    β[:,1] = m_n::Array{Float64,1}
    σ2[1] = mean((y .- x*m_n).^2)::Float64

    prog = Progress((iteration+burnin - 1), 1)
    for tt in 2:(iteration+burnin)
        # Use last iterations σ2
        σ2_tt = σ2[tt-1]

        # Sample β
        β_dist = MvNormal(m_n, Matrix(Hermitian(σ2_tt*V_n)))
        β_tt = rand(β_dist)

        # Sample σ2
        a_n = a_0 + (N/2) + (J+1)/2
        b_n = b_0 + 0.5*transpose((y .- x*β_tt))*(y .- x*β_tt) +
            0.5*transpose((β_tt - m_0_vec))*V_0_inv*(β_tt - m_0_vec)
        σ2_dist = InverseGamma(a_n, b_n)
        σ2_tt =  rand(σ2_dist)

        # Store parameter draws
        β[:,tt] = β_tt
        σ2[tt] = σ2_tt

        next!(prog)
    end

    return β[:,(burnin+1):(burnin+iteration)], σ2[(burnin+1):(burnin+iteration)]

end



"""
End of script
"""
