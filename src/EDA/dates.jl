using JSON3
using Plots
using DataStructures

include("plot_utils.jl")

function extract_dates_from_text(text::String)
    academic_year_pattern = r"\bcurso\s+(?:académico\s+)?(\d{4})[-/](\d{4})\b"i
    academic_years = collect(eachmatch(academic_year_pattern, text))
    
    recent_year_pattern = r"\b(202[0-9])\b"
    recent_years = unique([m.match for m in eachmatch(recent_year_pattern, text)])
    
    full_date_pattern = r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"
    full_dates = [m.match for m in eachmatch(full_date_pattern, text)]
    
    month_year_pattern = r"\b(xaneiro|febreiro|marzo|abril|maio|xuño|xullo|agosto|setembro|outubro|novembro|decembro|enero|febrero|junio|julio|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?(\d{4})\b"i
    month_years = collect(eachmatch(month_year_pattern, text))
    
    return (
        academic_years = [(m.captures[1], m.captures[2]) for m in academic_years],
        recent_years = recent_years,
        full_dates = full_dates,
        month_years = [(m.captures[1], m.captures[2]) for m in month_years]
    )
end

function analyze_content_dates(; solo_sin_lastmod=false)
    text_dir = "crawl/text"
    metadata_path = "crawl/metadata.json"
    
    if !isdir(text_dir) || !isfile(metadata_path)
        return
    end
    
    metadata = JSON3.read(read(metadata_path, String))
    
    docs_without_last_modified = Set{String}()
    for (url, meta) in pairs(metadata)
        if haskey(meta, :last_modified) && isnothing(meta.last_modified)
            if haskey(meta, :text_path)
                push!(docs_without_last_modified, basename(meta.text_path))
            end
        end
    end
    
    num_dates_hist = counter(Int)
    
    for file in filter(f -> endswith(f, ".txt"), readdir(text_dir))
        filepath = joinpath(text_dir, file)
        
        if solo_sin_lastmod && !(file in docs_without_last_modified)
            continue
        end
        
        try
            content = read(filepath, String)
            if length(content) > 50000
                content = content[1:50000]
            end
            
            dates = extract_dates_from_text(content)
            
            num_dates = (
                length(dates.academic_years) +
                length(dates.recent_years) +
                length(dates.full_dates) +
                length(dates.month_years)
            )
            
            inc!(num_dates_hist, num_dates)
        catch
        end
    end
    
    grouped_hist = counter(Int)
    for (count, freq) in num_dates_hist
        if count < 20
            grouped_hist[count] = freq
        else
            inc!(grouped_hist, 20, freq)
        end
    end
    
    x_vals = sort(collect(keys(grouped_hist)))
    y_vals = [grouped_hist[k] for k in x_vals]
    labels = [k < 20 ? string(k) : "20+" for k in x_vals]
    
    p = bar(
        labels, y_vals,
        color=COLOR_PRIMARY,
        xlabel="Número de datas no documento",
        ylabel="Número de documentos",
        title="Histograma: Datas extraídas por documento",
        legend=false,
        size=(800, 500)
    )
    
    add_value_labels!(p, 1:length(y_vals), y_vals)
    
    filename = solo_sin_lastmod ? "histograma_datas_sen_lastmod.png" : "histograma_datas.png"
    save_plot(p, filename)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        solo_null = "--solo-sin-lastmod" in ARGS
        analyze_content_dates(solo_sin_lastmod=solo_null)
    catch e
        println("ERROR in analyze_dates_in_content: $e")
    end
end
