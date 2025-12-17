using Plots
using Plots
using Colors
using Plots
using Colors
using LsqFit

function count_words(folder_path)
    word_counts = Dict{String, Int}()
    
    # Get all .txt files in the folder
    txt_files = filter(f -> endswith(f, ".txt"), readdir(folder_path))
    
    println("Found $(length(txt_files)) text files")
    
    for file in txt_files
        filepath = joinpath(folder_path, file)
        text = read(filepath, String)
        
        # Convert to lowercase and extract words (remove punctuation)
        words = split(lowercase(text), r"\W+")
        
        for word in words
            if !isempty(word)
                word_counts[word] = get(word_counts, word, 0) + 1
            end
        end
    end
    
    println("Total unique words: $(length(word_counts))")
    return word_counts
end

function rank_words(word_counts)
    # Sort by count (descending)
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
    # Initial guess: intercept and slope
    p0 = [log_counts[1], 1.0]
    
    # Fit linear model in log-log space
    fit = curve_fit(linear_model, log_ranks, log_counts, p0)
    
    return fit.param
end


function plot_corpus!(p, log_ranks, log_freqs, fitted_freqs, corpus_name, color, slope)
    # Add data points
    scatter!(
        p,
        log_ranks,
        log_freqs;
        label="$(corpus_name) - Data",
        markersize=3,
        color=color,
        alpha=0.45,
        markerstrokewidth=0
    )
    
    # Add fitted curve
    plot!(
        p,
        log_ranks,
        fitted_freqs;
        label="$(corpus_name) - Model (slope=$(round(slope, digits=2)))",
        linewidth=2.8,
        color=color,
        linestyle=:solid
    )
end

function create_zipf_plot()
    plot(
        xlabel="log(rank)",
        ylabel="log(relative frequency)",
        title="Word Frequency vs Rank - Multiple Corpora (Zipf's Law)",
        legend=:topright,
        grid=true,
        gridalpha=0.15,
        background_color=:white,
        foreground_color_subplot=:black,
        titlefont=font(14, "Helvetica"),
        guidefont=font(11),
        tickfont=font(9),
        legendfont=font(9),
        size=(900, 600)
    )
end

function create_parameter_plot(corpus_params)
    p = plot(
        xlabel="Intercept (a)",
        ylabel="Slope (b)",
        title="Zipf's Law Parameters by Corpus",
        legend=:best,
        grid=true,
        gridalpha=0.15,
        background_color=:white,
        foreground_color_subplot=:black,
        titlefont=font(14, "Helvetica"),
        guidefont=font(11),
        tickfont=font(9),
        legendfont=font(9),
        size=(700, 600)
    )
    
    # Plot each corpus as a point in parameter space
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
        
        # Add text annotation
        annotate!(p, a, b, 
		  text("  $(name)", 8, :left, color)
		  )
    end
    
    # Add reference line for ideal Zipf (slope = -1)
    #b_range = ylims(p)
    #hline!(p, [-1.0], linestyle=:dash, color=:gray, linewidth=2, label="Ideal Zipf (b=-1)")
    
    return p
end

"""
Print corpus statistics
"""
function print_corpus_stats(corpus_name, total_words, params, words, counts, relative_freqs)
    a, b = params
    println("Total words in corpus: $total_words")
    println("Fitted parameters: a = $(round(a, digits=3)), b = $(round(b, digits=3))")
    println("Top 5 words:")
    for j in 1:min(5, length(words))
        println("  $(j). $(words[j]): $(counts[j]) ($(round(relative_freqs[j]*100, digits=3))%)")
    end
end

function main()
    # Define corpora to analyze with colors
    corpora = [
        (folder="crawl/text", name="UDC", color=RGB(181/255, 60/255, 135/255)),
        (folder="corpora/crawl_ucm/text", name="UCM", color=RGB(60/255, 135/255, 181/255)),
        (folder="corpora/crawl_su/text", name="Stanford", color=RGB(135/255, 181/255, 60/255)),
        (folder="corpora/crawl_up/text", name="Paris", color=RGB(181/255, 135/255, 60/255)),
        (folder="corpora/brown", name="Brown", color=RGB(135/255, 60/255, 181/255)),
        (folder="corpora/quijote", name="Quijote", color=RGB(50/255, 168/255, 145/255))
    ]
    
    # Initialize plots
    zipf_plot = create_zipf_plot()
    
    # Store results for parameter plot
    corpus_results = []
    
    # Process each corpus
    for corpus in corpora
        println("\n" * "="^60)
        println("Processing corpus: $(corpus.name) ($(corpus.folder))")
        println("="^60)
        
        # Check if folder exists
        if !isdir(corpus.folder)
            println("Warning: Folder $(corpus.folder) does not exist. Skipping...")
            continue
        end
        
        word_counts = count_words(corpus.folder)
        
        if isempty(word_counts)
            println("Warning: No words found in $(corpus.folder). Skipping...")
            continue
        end
        
        total_words = sum(values(word_counts))
        
        ranks, words, counts = rank_words(word_counts)
        
        relative_freqs = counts ./ total_words
        
        log_ranks = log10.(ranks)
        log_freqs = log10.(relative_freqs)
        
        params = fit_model(log_ranks, log_freqs)
        
        
        fitted_freqs = linear_model(log_ranks, params)
        
        println("Step 5: Adding $(corpus.name) to plot...")
        plot_corpus!(
            zipf_plot,
            log_ranks,
            log_freqs,
            fitted_freqs,
            corpus.name,
            corpus.color,
            params[2]  # slope (b)
        )
        
        # Store for parameter plot
        push!(corpus_results, (corpus.name, params, corpus.color))
    end
    
    # Create parameter space plot
    param_plot = create_parameter_plot(corpus_results)
    
    
    zipf_output = "memoria/imaxes/zipf_modelo_normalized.png"
    param_output = "memoria/imaxes/zipf_parameters.png"
    
    savefig(zipf_plot, zipf_output)
    savefig(param_plot, param_output)
    
    println("Zipf plot saved to $zipf_output")
    println("Parameter plot saved to $param_output")
    println("="^60)
end

main()
