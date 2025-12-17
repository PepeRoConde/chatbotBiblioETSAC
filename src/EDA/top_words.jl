using DataStructures
using Plots
using CSV
using DataFrames

include("plot_utils.jl")

function read_text_files(directory="crawl/text/")
    text_parts = String[]
    
    for filename in readdir(directory)
        if endswith(filename, ".txt") && filename != "top_words.txt"
            filepath = joinpath(directory, filename)
            try
                push!(text_parts, read(filepath, String))
            catch
            end
        end
    end
    
    return join(text_parts, " ")
end

function extract_words(text)
    words = [m.match for m in eachmatch(r"\b[a-záéíóúñ]+\b", lowercase(text))]
    return words
end

function get_top_words(words, n)
    word_counts = counter(words)
    return sort(collect(word_counts), by=x->x[2], rev=true)[1:min(n+5, length(word_counts))]
end

function save_top_words_csv(top_words, n, output_file="crawl/top_words.csv")
    df = DataFrame(palabra=String[], frecuencia=Int[])
    
    for (i, (word, count)) in enumerate(top_words)
        if i <= n
            push!(df, (word, count))
        end
    end
    
    CSV.write(output_file, df)
end

function create_barplot(top_words, n)
    words = [word for (word, count) in top_words]
    counts = [count for (word, count) in top_words]
    
    colors = [i <= n ? COLOR_PRIMARY : COLOR_SECONDARY for i in 1:length(words)]
    
    p = bar(words, counts,
            color=colors,
            xlabel="Palabras",
            ylabel="Frecuencia",
            title="Top $n Palabras (+ 5 para contexto)",
            titlefontsize=14,
            legend=false,
            xrotation=45,
            size=(1200, 600))
    
    save_plot(p, "palabras.png")
end

function main(; n=15, directory="crawl")
    text_dir = joinpath(directory, "text")
    text = read_text_files(text_dir)
    
    if isempty(text)
        return
    end
    
    words = extract_words(text)
    top_words = get_top_words(words, n)
    
    if length(top_words) < n
        return
    end
    
    output_file = joinpath(directory, "top_words.txt")
    save_top_words_csv(top_words, n, output_file)
    
    create_barplot(top_words, n)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = 15
    directory = "crawl"
    
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "-n" && i < length(ARGS)
            global n = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] in ["-d", "--directory"] && i < length(ARGS)
            global directory = ARGS[i+1]
            i += 2
        else
            i += 1
        end
    end
    
    try
        main(n=n, directory=directory)
    catch e
        println("ERROR in top_words: $e")
    end
end
