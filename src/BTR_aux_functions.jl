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
Function that creates a box plot for regression coefficients
"""
function coef_plot(μ, σ, cmin, cmax; coef_no = 1, true_ω = "no")
    d = Normal(μ, σ)
    lo, hi = quantile.(d, [0.025, 0.975])
    mid_lo, mid_hi = quantile.(d, [0.25, 0.75])
    plt =  plot(xlim = (cmin, cmax), ylim = (0.0, 0.1), yticks=false)
    plot!([lo,hi], [0.05, 0.05], color = :dodgerblue)
    plot!([mid_lo, mid_hi], [0.075, 0.075], color = :lightskyblue2,
        fillrange=[0.025 0.075], fillalpha = 0.5)
    plot!([mid_lo, mid_hi], [0.025, 0.025], color = :lightskyblue2)
    plot!([μ, μ], [0.02, 0.08], color = :blue)
    plot!([lo, lo], [0.04, 0.06], color = :dodgerblue)
    plot!([hi, hi], [0.04, 0.06], color = :dodgerblue)
    plot!(legend = false, xlabel = join(["Topic ", string(coef_no)]))

    if true_ω != "no"
        scatter!([true_ω], [0.05], color = :red)
    end
    return plt
end


"""
Function to re-order synthetic data output
"""
function synth_reorder_topics(β)
    order = [0, 0, 0]
    for kk in 1:3
        indx = findmax(β[kk,:])[2]
        if indx < 4
            order[1] = kk
        elseif indx >6
            order[3] = kk
        else
            order[2] = kk
        end
    end
    return order
end


"""
Function that plots BTR output for synthetic data (currently only works for 3 topics)
    topic_ord re orders β, e.g. [3,2,1] would convert topic 3 to 1, and topic 1 to 3 (with 2 staying as 2)
"""
function synth_data_plot(β,ω,Σ; true_ω = "no", topic_ord = [1,2,3])
    # Re-order β
    β = β[topic_ord,:]
    cmin = minimum(ω .- maximum(vcat(0.1, 2.5*sqrt.(diag(Σ)))))
    cmax = maximum(ω .+ maximum(vcat(0.1, 2.5*sqrt.(diag(Σ)))))
    if true_ω == "no"
        coef_true = fill("no", length(ω))
    else
        coef_true = true_ω
        cmin = minimum(vcat(cmin, true_ω .- 0.1))
        cmax = maximum(vcat(cmax, true_ω .+ 0.1))

    end
    # Define plot layout
    l = @layout [
        a{0.6w} [b{0.33h}
                 c{0.33h}
                 d{0.33h} ]]
    # Plot the estimated topics
    p1 = heatmap(β, xlabel = "Vocab", ylabel = "Topic", yticks = 1:K)
    # Plot regression coefficient for each topic
    p4 = coef_plot(ω[topic_ord[1]], sqrt(diag(Σ)[topic_ord[1]]), cmin, cmax, coef_no = 1, true_ω = coef_true[1])
    p3 = coef_plot(ω[topic_ord[2]], sqrt(diag(Σ)[topic_ord[2]]), cmin, cmax, coef_no = 2, true_ω = coef_true[2])
    p2 = coef_plot(ω[topic_ord[3]], sqrt(diag(Σ)[topic_ord[3]]), cmin, cmax, coef_no = 3, true_ω = coef_true[3])
    plt = plot(p1, p2, p3, p4, layout = l)
    return plt
end
# Afternative function if we have an empirical distribution rather than mean and variance
function synth_data_plot(β::Array{Float64,2}, ω_post::Array{Float64,2};
    true_ω = "no", topic_ord = [1,2,3], plt_title::String = "", legend = false,
    interactions = Array{Int64,1}([]), top_mar::Int = 0,
    left_mar::Int = 0, bottom_mar::Int = 0,
    ticksize::Int = 6, labelsize::Int = 12)
    # Extract key info
    n_draws = size(ω_post,2)
    ntopics = size(β,1)
    ncoefs = size(ω_post,1)
    # Re-order β
    β = β[topic_ord,:]
    ω_post = ω_post[topic_ord,:]
    cmin = minimum(ω_post) - 0.25
    cmax = maximum(ω_post) + 0.25
    if true_ω == "no"
        coef_true = fill("no", length(ω))
    else
        true_ω = true_ω
        cmin = minimum(vcat(cmin, true_ω .- 0.1))
        cmax = maximum(vcat(cmax, true_ω .+ 0.1))

    end
    # Define plot layout
    l = @layout [
        a{0.5w} [b{0.33h}
                 c{0.33h}
                 d{0.33h} ]]
    # Plot the estimated topics
    p1 = heatmap(β, xlabel = "Vocab", ylabel = "Topic", yticks = 1:ntopics,
        title= plt_title, legend = legend, left_margin = Int(left_mar)mm,
        top_margin = Int(top_mar)mm, bottom_margin = Int(bottom_mar)mm)
    plot!(xtickfontize = ticksize,ytickfontize = ticksize)
    plts=[plot(),plot(),plot()]
    # Plot regression coefficient for each topic
    for kk in 1:ntopics
        ptemp=plot(legend=false,plot(xlim = (cmin, cmax), yticks=false))
        ω_kk = sort(ω_post[kk,:])
        lo = ω_kk[Int(round(0.025*n_draws))]
        hi = ω_kk[Int(round(0.975*n_draws))]
        mid_lo = ω_kk[Int(round(0.25*n_draws))]
        mid_hi =  ω_kk[Int(round(0.75*n_draws))]
        mid = mean(ω_kk)
        plot!([lo,hi], [kk, kk], color = :dodgerblue)
        plot!([mid_lo, mid_hi], [kk+0.2, kk+0.2], color = :lightskyblue2,
        fillrange=[kk-0.2, kk-0.2], fillalpha = 0.5)
        plot!([mid_lo, mid_hi], [kk-0.2, kk-0.2], color = :lightskyblue2)
        plot!([mid, mid], [kk-0.4, kk+0.4], color = :blue)
        plot!([lo, lo], [kk-0.1, kk+0.1], color = :dodgerblue)
        plot!([hi, hi], [kk-0.1, kk+0.1], color = :dodgerblue)
        if true_ω != "no"
            scatter!([true_ω[kk]], [kk], color = :red)
        end
        plot!(xtickfontize = ticksize,ytickfontize = ticksize)
        if kk>1
            plot!(xticklabel = false)
        else
            plot!(xlabel = "Coefficient")
        end

        # for int in 1:length(interactions)
        #     coef_range = ntopics+ntopics*int-ntopics+1:ntopics+ntopics*int
        #     ω_ints = ω_post[coef_range,:]
        #     ω_int = sort(ω_ints[kk,:])
        #     lo = ω_int[Int(round(0.025*n_draws))]
        #     hi = ω_int[Int(round(0.975*n_draws))]
        #     mid_lo = ω_int[Int(round(0.25*n_draws))]
        #     mid_hi =  ω_int[Int(round(0.75*n_draws))]
        #     mid = mean(ω_int)
        #     plot!([lo,hi], [kk, kk], color = :darkolivegreen1)
        #     plot!([mid_lo, mid_hi], [kk+0.2, kk+0.2], color = :darkseagreen1,
        #     fillrange=[kk-0.2, kk-0.2], fillalpha = 0.5)
        #     plot!([mid_lo, mid_hi], [kk-0.2, kk-0.2], color = :darkseagreen1)
        #     plot!([mid, mid], [kk-0.4, kk+0.4], color = :olivedrab1)
        #     plot!([lo, lo], [kk-0.1, kk+0.1], color = :darkolivegreen1)
        #     plot!([hi, hi], [kk-0.1, kk+0.1], color = :darkolivegreen1)
        #     if true_ω != "no"
        #         scatter!([true_ω[coef_range[kk]]], [kk], color = :red)
        #     end
        #
        # end
        plts[kk]=ptemp
    end

    plt = plot(p1, plts[3], plts[2], plts[1], layout = l)
    #plt = plot(p1, plot(), plot(), plot(), layout = l)
    plot!(xtickfontsize=ticksize)
    plot!(ytickfontsize=ticksize)
    plot!(xlabelfontsize=labelsize)
    plot!(ylabelfontsize=labelsize)
    return plt

end




"""
Plot top words and coefficients for BLR output
"""
function BTR_plot(β::Array{Float64,2}, ω::Array{Float64,1}, Σ::Array{Float64,2};
    nwords::Int64 = 10, plt_title::String = "", fontsize::Int64 = 6)
    # Extract key info
    σ = sqrt.(diag(Σ))
    ntopics = size(β,1)

    top_words = DataFrame(string.(zeros(nwords, ntopics)))
    rename!(top_words, [Symbol("T$kk") for kk in 1:ntopics])

    topic_df = DataFrame(transpose(β))
    rename!(topic_df, [Symbol("T$kk") for kk in 1:ntopics])
    topic_df.term = deepcopy(vocab)
    ytick_words = String[]
    for kk in 1:ntopics
        temp_df = last(sort!(topic_df, [kk], rev = false), nwords)
        top_words[:,kk] = temp_df.term
        push!(ytick_words, join(temp_df.term, "."))
    end

    cmin = minimum(ω .- maximum(vcat(0.25, 2.5*sqrt.(diag(Σ)))))
    cmax = maximum(ω .+ maximum(vcat(0.25, 2.5*sqrt.(diag(Σ)))))

    plt = plot(xguidefontsize=8, left_margin = 7mm, legend = false,
        ylim = (0,ntopics+1), xlim = (cmin, cmax),
        xlabel = "Coefficient", ylabel = "Topic",
        yticks = (1:ntopics, ytick_words),ytickfontsize=fontsize)

    # Plot regression coefficient for each topic
    for kk in 1:ntopics
        d = Normal(ω[kk], σ[kk])
        lo, hi = quantile.(d, [0.025, 0.975])
        mid_lo, mid_hi = quantile.(d, [0.25, 0.75])
        plot!([lo,hi], [kk, kk], color = :dodgerblue)
        plot!([mid_lo, mid_hi], [kk+0.2, kk+0.2], color = :lightskyblue2,
        fillrange=[kk-0.2, kk-0.2], fillalpha = 0.5)
        plot!([mid_lo, mid_hi], [kk-0.2, kk-0.2], color = :lightskyblue2)
        plot!([ω[kk], ω[kk]], [kk-0.4, kk+0.4], color = :blue)
        plot!([lo, lo], [kk-0.1, kk+0.1], color = :dodgerblue)
        plot!([hi, hi], [kk-0.1, kk+0.1], color = :dodgerblue)
    end
    plot!(title = plt_title)

    return plt
end


# Afternative function if we have an empirical distribution rather than mean and variance
function BTR_plot(β::Array{Float64,2}, ω_post::Array{Float64,2};
    nwords::Int64 = 10, plt_title::String = "", left_mar::Int64 = 7,
    fontsize::Int64 = 6, title_size::Int64 = 14,
    interactions::Array{Int64,1}=Array{Int64,1}([]))
    # Extract key info
    ntopics = size(β,1)

    top_words = DataFrame(string.(zeros(nwords, ntopics)))
    rename!(top_words, [Symbol("T$kk") for kk in 1:ntopics])

    topic_df = DataFrame(transpose(β))
    rename!(topic_df, [Symbol("T$kk") for kk in 1:ntopics])
    topic_df.term = deepcopy(vocab)
    ytick_words = String[]
    for kk in 1:ntopics
        temp_df = last(sort!(topic_df, [kk], rev = false), nwords)
        top_words[:,kk] = temp_df.term
        push!(ytick_words, join(temp_df.term, "."))
    end

    #cmin = minimum(ω_post) - 0.25
    #cmax = maximum(ω_post) + 0.25

    plt1 = plot(xguidefontsize=8, left_margin = Int(left_mar)mm, legend = false,
        ylim = (0,ntopics+1),
        xlabel = "Coefficient", ylabel = "Topic",
        yticks = (1:ntopics, ytick_words),ytickfontsize=fontsize,
        xtickfontsize = fontsize)
    plot!([0.,0.],[0.,(Float64(ntopics)+0.6)],linestyle = :dash,color =:red)

    # Plot regression coefficient for each topic
    for kk in 1:ntopics
        ω_kk = sort(ω_post[kk,:])
        plt1
        coef_plot(ω_kk,kk,scheme = :blue)
    end
    plot!(title = plt_title, titlefontsize= title_size)

    # Plot interaction coefficients
    if length(interactions) == 1
        plt2 = plot(xguidefontsize=8, left_margin = Int(left_mar)mm, legend = false,
            ylim = (0,ntopics+1),
            xlabel = "Interaction",
            yticks = false)
        plot!([0.,0.],[0.,(Float64(ntopics)+0.6)],linestyle = :dash,color =:red)

        for kk in 1:ntopics
            ω_kk = sort(ω_post[kk,:])
            if length(interactions) == 1
                ω_kx = sort(ω_post[interactions[1]*ntopics+kk,:])
                coef_plot(ω_kx,kk,scheme = :green)
            end
        end
    end

    # Combine if necessary
    if length(interactions) == 1
        l = @layout [
            a{0.5w} b{0.5w}]
        plt = plot(plt1, plt2, layout = l)
    else
        plt = plt1
    end

    return plt
end


"""
Function that makes box plots for a coefficient from sampled values
"""
function coef_plot(ω_kk,kk;scheme = :blue)

    n_draws = length(ω_kk)

    if scheme == :blue
        col_pal = ColorSchemes.dense
    elseif scheme == :green
        col_pal = ColorSchemes.algae
    else
        error("Need to specify valid colour scheme")
    end

    # Find intervals
    lo = ω_kk[Int(round(0.025*n_draws))]
    hi = ω_kk[Int(round(0.975*n_draws))]
    mid_lo = ω_kk[Int(round(0.25*n_draws))]
    mid_hi =  ω_kk[Int(round(0.75*n_draws))]
    mid = mean(ω_kk)

    # Plot box and whiskers
    plot!([lo,hi], [kk, kk], color = col_pal[50])
    plot!([mid_lo, mid_hi], [kk+0.2, kk+0.2], color = col_pal[30],
        fillrange=[kk-0.2, kk-0.2], fillalpha = 0.5)
    plot!([mid_lo, mid_hi], [kk-0.2, kk-0.2], color = col_pal[50])
    plot!([mid, mid], [kk-0.4, kk+0.4], color = col_pal[100])
    plot!([lo, lo], [kk-0.1, kk+0.1], color = col_pal[50])
    plot!([hi, hi], [kk-0.1, kk+0.1], color = col_pal[50])
end




"""
Plot topics from a β matrix and vocab array
"""
function plot_topics(β, vocab, ω)
    ntopics = size(β,1)

    topic_df = DataFrame(transpose(β))
    rename!(topic_df, [Symbol("T$kk") for kk in 1:ntopics])
    topic_df.term = deepcopy(vocab)
    plt = plot(layout = (Int(ntopics/2),2),xguidefontsize=8, size = (1000,1000), left_margin = 6mm)

    for kk in 1:ntopics
        temp_df = last(sort!(topic_df, [kk], rev = false), 10)
        if kk < ntopics
            bar!(temp_df.term, temp_df[:,kk], orientation = :horizontal, subplot = kk, legend = false,
            title = join(["T", kk, ": ", round(ω[kk],digits=2)]))
        else
            display(bar!(temp_df.term, temp_df[:,kk], orientation = :horizontal, subplot = kk, legend = false,
            title = join(["T", kk, ": ", round(ω[kk],digits=2)])))
        end
    end
    return plt
end





"""
Closed form Bayesian Linear Regression with known σ
"""
function BLR(y::Array{Float64,1},x::Array{Float64,2},
    μ::Array{Float64,1},S::Array{Float64,2},σ2::Float64)
    Σ_inv = σ2^(-1)*(transpose(x)*x) .+ inv(S)
    m = σ2^(-1)*inv(Σ_inv)*(transpose(x)*y)
    Σ = inv(Σ_inv)
    return m,Σ
end



"""
Gibbs sampled Bayesian Linear Regression with NIG priors
    μ_0 is the mean of the coefficient prior
    S_0 is the covariance matrix of the coefficient prior
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
