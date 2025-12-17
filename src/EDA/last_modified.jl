using JSON3
using Dates
using Plots
using DataStructures

include("plot_utils.jl")

function parse_last_modified(date_str::String)
    try
        return DateTime(date_str, dateformat"e, d u y H:M:S \G\M\T")
    catch
        return nothing
    end
end

function analyze_last_modified()
    metadata_path = "crawl/metadata.json"
    
    if !isfile(metadata_path)
        return
    end
    
    metadata = JSON3.read(read(metadata_path, String))
    
    total_docs = length(metadata)
    with_last_modified = 0
    without_last_modified = 0
    dates_by_month = counter(String)
    dates_by_year = counter(String)
    
    for (url, meta) in pairs(metadata)
        last_modified = get(meta, :last_modified, nothing)
        
        if !isnothing(last_modified)
            with_last_modified += 1
            
            dt = parse_last_modified(last_modified)
            if !isnothing(dt)
                month_key = Dates.format(dt, "yyyy-mm")
                inc!(dates_by_month, month_key)
                
                year_key = Dates.format(dt, "yyyy")
                inc!(dates_by_year, year_key)
            end
        else
            without_last_modified += 1
        end
    end
    
    p1 = create_pie_chart(with_last_modified, without_last_modified)
    p2 = create_year_bar_chart(dates_by_year)
    p3 = create_month_line_chart(dates_by_month)
    p4 = create_stats_table(total_docs, with_last_modified, without_last_modified, dates_by_year)
    
    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1400, 1000),
             plot_title="Análise de Datas Last-Modified nos Documentos",
             plot_titlefontsize=16)
    
    save_plot(p, "last_modified_analysis.png")
end

function create_pie_chart(with_date, without_date)
    labels = ["Con Last-Modified\n($with_date)", "Sen Last-Modified\n($without_date)"]
    sizes = [with_date, without_date]
    colors = [RGB(76/255, 175/255, 80/255), RGB(255/255, 82/255, 82/255)]
    
    pie(sizes, 
        labels=labels,
        colors=colors,
        title="Dispoñibilidade de Last-Modified",
        titlefontsize=12,
        legend=false)
end

function create_year_bar_chart(by_year)
    if isempty(by_year)
        return plot(
            title="Documentos por Ano (Last-Modified)",
            titlefontsize=12,
            annotations=(0.5, 0.5, "Sen datos de ano dispoñibles")
        )
    end
    
    years = sort(collect(keys(by_year)))
    counts = [by_year[y] for y in years]
    
    p = bar(years, counts,
            color=COLOR_ACCENT1,
            alpha=0.8,
            xlabel="Ano",
            ylabel="Cantidade de documentos",
            title="Documentos por Ano (Last-Modified)",
            titlefontsize=12,
            legend=false)
    
    add_value_labels!(p, 1:length(counts), counts)
    return p
end

function create_month_line_chart(by_month)
    if isempty(by_month)
        return plot(
            title="Evolución Temporal (últimos 12 meses)",
            titlefontsize=12,
            annotations=(0.5, 0.5, "Sen datos mensuais dispoñibles")
        )
    end
    
    recent_months = sort(collect(keys(by_month)))[end-min(11, length(by_month)-1):end]
    counts = [by_month[m] for m in recent_months]
    
    p = plot(1:length(recent_months), counts,
            marker=:circle,
            markersize=8,
            linewidth=2,
            color=RGB(255/255, 152/255, 0/255),
            markerfacecolor=RGB(255/255, 193/255, 7/255),
            xlabel="Mes",
            ylabel="Cantidade de documentos",
            title="Evolución Temporal (últimos 12 meses)",
            titlefontsize=12,
            legend=false,
            xticks=(1:length(recent_months), recent_months),
            xrotation=45)
    
    for (i, count) in enumerate(counts)
        annotate!(p, i, count, text(string(count), 8, :bottom))
    end
    
    return p
end

function create_stats_table(total, with_date, without_date, by_year)
    stats = String[]
    push!(stats, "ESTATÍSTICAS DETALLADAS")
    push!(stats, "")
    push!(stats, "Total documentos: $total")
    push!(stats, "Con Last-Modified: $with_date ($(round(with_date/total*100, digits=1))%)")
    push!(stats, "Sen Last-Modified: $without_date ($(round(without_date/total*100, digits=1))%)")
    push!(stats, "")
    
    coverage = with_date / total
    status = coverage > 0.7 ? "✅ Boa" : coverage > 0.4 ? "⚠️ Regular" : "❌ Baixa"
    push!(stats, "Cobertura temporal: $status")
    
    if !isempty(by_year)
        oldest_year = minimum(keys(by_year))
        newest_year = maximum(keys(by_year))
        push!(stats, "Rango temporal: $oldest_year - $newest_year")
    end
    
    p = plot(
        xlims=(0, 1),
        ylims=(0, 1),
        title="Estatísticas Detalladas",
        titlefontsize=12,
        legend=false,
        grid=false,
        showaxis=false,
        ticks=false
    )
    
    y_pos = 0.9
    for line in stats
        annotate!(p, 0.1, y_pos, text(line, 10, :left, :black))
        y_pos -= 0.12
    end
    
    return p
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        analyze_last_modified()
    catch e
        println("ERROR in analyze_last_modified: $e")
    end
end
