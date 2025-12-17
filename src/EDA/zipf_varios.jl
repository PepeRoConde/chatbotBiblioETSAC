using Plots
using Colors
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
    log_ranks = log10.(ranks)
    log_counts = log10.(counts)
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

function plot_corpus!(p, log_ranks, log_freqs, fitted_freqs, corpus_name, color, slope)
    scatter!(
        p,
        log_ranks,
        log_freqs;
        label="$(corpus_name) - Datos",
        markersize=3,
        color=color,
        alpha=0.45,
        markerstrokewidth=0
    )
    
    plot!(
        p,
        log_ranks,
        fitted_freqs;
        label="$(corpus_name) - Modelo (pendente=$(round(slope, digits=2)))",
        linewidth=2.8,
        color=color,
        linestyle=:solid
    )
end

function create_parameter_plot(corpus_params)
    p = plot(
        xlabel="Intercepto (a)",
        ylabel="Pendente (b)",
        title="Parámetros da Lei de Zipf por Corpus",
        legend=:best,
        size=(700, 600)
    )
    
    for (i, (name, params, color)) in enumerate(corpus_params)
        a, b = params
        scatter!(
            p,
            [a],
            [b];
            label=name,
            markersize=10,
            color=color,
            markerstrokewidth=2,
            markerstrokecolor=:white
        )
        
        annotate!(p, a, b, text("  $(name)", 8, :left, color))
    end
    
    return p
end

function main()
    corpora = [
        (folder="crawl/text", name="UDC", color=COLOR_PRIMARY),
        (folder="corpora/crawl_ucm/text", name="UCM", color=COLOR_ACCENT1),
        (folder="corpora/crawl_su/text", name="Stanford", color=COLOR_ACCENT2),
        (folder="corpora/crawl_up/text", name="París", color=COLOR_ACCENT3),
        (folder="corpora/brown", name="Brown", color=COLOR_ACCENT4),
        (folder="corpora/quijote", name="Quixote", color=COLOR_ACCENT5)
    ]
    
    zipf_plot = plot(
        xlabel="log(rango)",
        ylabel="log(frecuencia relativa)",
        title="Frecuencia de Palabras vs Rango - Múltiples Corpus (Lei de Zipf)",
        legend=:topright,
        size=(900, 600)
    )
    
    corpus_results = []
    
    for corpus in corpora
        if !isdir(corpus.folder)
            continue
        end
        
        word_counts = count_words(corpus.folder)
        
        if isempty(word_counts)
            continue
        end
        
        total_words = sum(values(word_counts))
        ranks, words, counts = rank_words(word_counts)
        relative_freqs = counts ./ total_words
        log_ranks = log10.(ranks)
        log_freqs = log10.(relative_freqs)
        params = fit_model(log_ranks, log_freqs)
        fitted_freqs = linear_model(log_ranks, params)
        
        plot_corpus!(zipf_plot, log_ranks, log_freqs, fitted_freqs, corpus.name, corpus.color, params[2])
        push!(corpus_results, (corpus.name, params, corpus.color))
    end
    
    param_plot = create_parameter_plot(corpus_results)
    
    save_plot(zipf_plot, "zipf_modelo_normalizado.png")
    save_plot(param_plot, "zipf_parametros.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch e
        println("ERROR in zipf_varios: $e")
    end
end
