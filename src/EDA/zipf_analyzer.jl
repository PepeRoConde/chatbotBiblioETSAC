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
    folder_path = "crawl/text"

    
    println("Step 1: Counting words...")
    word_counts = count_words(folder_path)
    
    println("Step 2: Ranking words...")
    ranks, words, counts = rank_words(word_counts)
    
    println("Step 3: Converting to log space...")
    log_ranks, log_counts = to_log_space(ranks, counts)
    
    println("Step 4: Fitting model for $(folder_path)")
    params = fit_model(log_ranks, log_counts)
    a, b = params
    println("Fitted parameters: a = $a, b = $b")
    
    fitted_counts = linear_model(log_ranks, params)
    
    println("Step 5: Creating plot...")
    model_color = RGB(181/255, 60/255, 135/255)
    
    p = scatter(
        log_ranks,
        log_counts;
        label="Data",
        xlabel="log(rank)",
        ylabel="log(word count)",
        title="Word Frequency vs Rank (Zipf's Law)",
        markersize=3,
        color=:gray60,
        alpha=0.45,
        markerstrokewidth=0,
        legend=:topright,
        grid=true,
        gridalpha=0.15,
        background_color=:white,
        foreground_color_subplot=:black,
        titlefont=font(14, "Helvetica"),
        guidefont=font(11),
        tickfont=font(9),
        legendfont=font(9),
    )
    
    # Plot fitted curve
    plot!(
        log_ranks,
        fitted_counts;
        label="Fitted model",
        linewidth=2.8,
        color=model_color,
    )
    
    # Annotate specific words
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
    
    # Add parameter text
    param_text = "a = $(round(a, digits=3))\nb = $(round(b, digits=3))"
    annotate!(
        minimum(log_ranks) + 0.05 * (maximum(log_ranks) - minimum(log_ranks)),
        minimum(log_counts) + 0.1 * (maximum(log_counts) - minimum(log_counts)),
        text(param_text, 10, model_color, :left)
    )
 
    display(p)
 
    output_file = "memoria/imaxes/zipf_modelo.png"
    savefig(p, output_file)
    println("Plot saved to $output_file")
end

# Run the program
main()
