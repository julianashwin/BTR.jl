"""
This file defines functions to visualise the output of Bayesian Topic Regressions
"""


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
function BTR_plot(β::Array{Float64,2}, ω_post::Array{Float64,2}, vocab::Array{String,1};
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
        temp_df = first(sort!(topic_df, [kk], rev = true), nwords)
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
End of script
"""
