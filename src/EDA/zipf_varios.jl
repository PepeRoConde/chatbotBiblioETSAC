using Plots
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

# Main execution
function main()
    # Define corpora to analyze
    corpora = [
        (folder="crawl/text", name="UDC"),
	(folder="corpora/crawl_ucm/text", name="UCM"),
        (folder="corpora/crawl_su/text", name="Standford"),
        (folder="corpora/crawl_up/text", name="Paris"),
        (folder="corpora/brown", name="Brown"),
        (folder="corpora/quijote", name="Quijote")
    ]
    
    # Color palette for different corpora
    colors = [
        RGB(181/255, 60/255, 135/255),  # Pink
        RGB(60/255, 135/255, 181/255),  # Blue
        RGB(135/255, 181/255, 60/255),  # Green
        RGB(181/255, 135/255, 60/255),  # Orange
        RGB(135/255, 60/255, 181/255),   # Purple
        RGB(50/255, 168/255, 145/255),   # Purple
        RGB(50/255, 97/255, 30/255)   # Purple
    ]
    
    # Initialize plot
    p = plot(
        xlabel="log(rank)",
        ylabel="log(word count)",
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
    
    # Process each corpus
    for (i, corpus) in enumerate(corpora)
        println("\n" * "="^60)
        println("Processing corpus: $(corpus.name) ($(corpus.folder))")
        println("="^60)
        
        # Check if folder exists
        if !isdir(corpus.folder)
            println("Warning: Folder $(corpus.folder) does not exist. Skipping...")
            continue
        end
        
        println("Step 1: Counting words...")
        word_counts = count_words(corpus.folder)
        
        if isempty(word_counts)
            println("Warning: No words found in $(corpus.folder). Skipping...")
            continue
        end
        
        # Calculate total words for normalization
        total_words = sum(values(word_counts))
        println("Total words in corpus: $total_words")
        
        println("Step 2: Ranking words...")
        ranks, words, counts = rank_words(word_counts)
        
        # Normalize counts: convert to relative frequencies
        relative_freqs = counts ./ total_words
        
        println("Step 3: Converting to log space...")
        log_ranks = log10.(ranks)
        log_freqs = log10.(relative_freqs)
        
        println("Step 4: Fitting model for $(corpus.name)")
        params = fit_model(log_ranks, log_freqs)
        a, b = params
        println("Fitted parameters: a = $(round(a, digits=3)), b = $(round(b, digits=3))")
        
        fitted_freqs = linear_model(log_ranks, params)
        
        # Get color for this corpus
        color = colors[mod1(i, length(colors))]
        
        println("Step 5: Adding to plot...")
        
        # Add data points
        scatter!(
            p,
            log_ranks,
            log_freqs;
            label="$(corpus.name) - Data",
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
            label="$(corpus.name) - Model (slope=$(round(b, digits=2)))",
            linewidth=2.8,
            color=color,
            linestyle=:solid
        )
        
        # Print some statistics
        println("Top 5 words:")
        for j in 1:min(5, length(words))
            println("  $(j). $(words[j]): $(counts[j]) ($(round(relative_freqs[j]*100, digits=3))%)")
        end
    end
    
    println("\n" * "="^60)
    println("Displaying combined plot...")
    display(p)
    
    output_file = "memoria/imaxes/zipf_modelo_normalized.png"
    savefig(p, output_file)
    println("Plot saved to $output_file")
    println("="^60)
end

# Run the program
main()
