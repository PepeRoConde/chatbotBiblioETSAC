using Plots
using LsqFit
using DataStructures

include("plot_utils.jl")

function count_words(folder_path)
    word_counts = counter(String)
    txt_files = filter(f -> endswith(f, ".txt"), readdir(folder_path))
    
    for file in txt_files
        filepath = joinpath(folder_path, file)
        text = read(filepath, String)
        words = split(lowercase(text), r"\W+")
        
        for word in words
            if !isempty(word)
                inc!(word_counts, word)
            end
        end
    end
    
    return word_counts
end

function rank_words(word_counts)
    sorted_words = sort(collect(word_counts), by=x->x[2], rev=true)
    
    ranks = collect(1:length(sorted_words))
    words = [w[1] for w in sorted_words]
    counts = [w[2] for w in sorted_words]
    
    return ranks, words, counts
end

function to_log_space(ranks, counts)
    log_ranks = log.(ranks)
    log_counts = log.(counts)
    return log_ranks, log_counts
end

function linear_model(log_rank, params)
    log_C, alpha = params
    return log_C .- alpha .* log_rank
end

function fit_model(log_ranks, log_counts)
    p0 = [log_counts[1], 1.0]
    fit = curve_fit(linear_model, log_ranks, log_counts, p0)
    return fit.param
end

function main()
    folder_path = "crawl/text"
    
    word_counts = count_words(folder_path)
    ranks, words, counts = rank_words(word_counts)
    log_ranks, log_counts = to_log_space(ranks, counts)
    params = fit_model(log_ranks, log_counts)
    a, b = params
    
    fitted_counts = linear_model(log_ranks, params)
    
    p = scatter(
        log_ranks,
        log_counts;
        label="Datos",
        xlabel="log(rango)",
        ylabel="log(conteo de palabras)",
        title="Frecuencia de Palabras vs Rango (Lei de Zipf)",
        markersize=3,
        color=:gray60,
        alpha=0.45,
        markerstrokewidth=0,
        legend=:topright,
        size=(900, 600)
    )
    
    plot!(
        log_ranks,
        fitted_counts;
        label="Modelo axustado",
        linewidth=2.8,
        color=COLOR_PRIMARY,
    )
    
    annotate_ranks = [1, 10, 25, 75, 150, 300, 500, 1000, 2000]
    for r in annotate_ranks
        if r <= length(words)
            idx = findfirst(==(r), ranks)
            if idx !== nothing
                annotate!(
                    log_ranks[idx],
                    log_counts[idx],
                    text("  $(words[idx])", 8, :left, :gray20)
                )
            end
        end
    end
    
    param_text = "a = $(round(a, digits=3))\nb = $(round(b, digits=3))"
    x_pos = minimum(log_ranks) + 0.05 * (maximum(log_ranks) - minimum(log_ranks))
    y_pos = minimum(log_counts) + 0.1 * (maximum(log_counts) - minimum(log_counts))
    annotate!(x_pos, y_pos, text(param_text, 10, COLOR_PRIMARY, :left))
 
    save_plot(p, "zipf_modelo.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch e
        println("ERROR in zipf_analyzer: $e")
    end
end
