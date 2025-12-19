using CSV
using DataFrames
using Plots
using Statistics

include("plot_utils.jl")

function plot_ocr_statistics(csv_path="crawl/ocr_stats.csv")
    if !isfile(csv_path)
        println("ERROR: CSV file not found: $csv_path")
        return 1
    end
    
    try
        df = CSV.read(csv_path, DataFrame)
        
        # Drop rows with any missing values
        df_clean = dropmissing(df)
        
        if nrow(df_clean) == 0
            println("ERROR: No valid rows after removing missing values")
            return 1
        end
        
        df_valid = filter(row -> row.char_count > 0, df_clean)
        
        if nrow(df_valid) == 0
            println("ERROR: No rows with char_count > 0")
            return 1
        end
        
        p1 = create_table_pie_chart(df_valid)
        p2 = create_char_distribution_plot(df_valid)
        
        p = plot(p1, p2, layout=(1, 2), size=(1400, 600))
        
        save_plot(p, "ocr.png")
        
        return 0
    catch e
        println("ERROR in plot_ocr_statistics: $e")
        return 1
    end
end

function create_table_pie_chart(df_valid)
    table_counts = combine(groupby(df_valid, :is_table), nrow => :count)
    sort!(table_counts, :is_table)
    
    labels = ["Non Táboa", "Táboa"]
    sizes = table_counts.count
    colors = [RGB(255/255, 153/255, 153/255), COLOR_PRIMARY]
    
    pie(sizes,
        labels=labels,
        colors=colors,
        title="Imaxes: Táboas vs Non-Táboas",
        titlefontsize=14,
        legend=false,
        startangle=90)
end

function create_char_distribution_plot(df_valid)
    non_tables = filter(row -> !row.is_table, df_valid)
    
    if nrow(non_tables) == 0
        return plot(
            title="Imaxes Non-Táboa: Distribución de Caracteres",
            titlefontsize=14,
            annotations=(0.5, 0.5, "Non se atoparon imaxes non-táboa")
        )
    end
    
    sort!(non_tables, :char_count)
    char_counts = non_tables.char_count
    frequencies = 1:length(char_counts)
    
    p = plot(char_counts, frequencies,
            marker=:circle,
            markersize=4,
            linewidth=2,
            color=COLOR_PRIMARY,
            xlabel="Cantidade de caracteres",
            ylabel="Frecuencia (Acumulativa)",
            title="Imaxes Non-Táboa: Distribución de Caracteres",
            titlefontsize=14,
            legend=false)
    
    stats_text = "Mín: $(minimum(char_counts))\nMáx: $(maximum(char_counts))\nMedia: $(round(mean(char_counts), digits=1))"
    
    x_pos = minimum(char_counts) + 0.95 * (maximum(char_counts) - minimum(char_counts))
    y_pos = minimum(frequencies) + 0.05 * (maximum(frequencies) - minimum(frequencies))
    
    annotate!(p, x_pos, y_pos, text(stats_text, 9, :right, :black, :box, :wheat, 0.5))
    
    return p
end

if abspath(PROGRAM_FILE) == @__FILE__
    csv_path = length(ARGS) > 0 ? ARGS[1] : "crawl/ocr_stats.csv"
    exit(plot_ocr_statistics(csv_path))
end
