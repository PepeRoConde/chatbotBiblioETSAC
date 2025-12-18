using CSV
using DataFrames
using Plots
using Statistics

include("plot_utils.jl")

function load_ocr_data(csv_path="crawl/ocr_stats.csv")
    if !isfile(csv_path)
        error("CSV file not found: $csv_path")
    end
    
    # Leer CSV con manejo de errores
    df = CSV.read(csv_path, DataFrame, 
                  silencewarnings=true,
                  strict=false)
    
    # Eliminar filas con valores missing en columnas críticas
    df = dropmissing(df, [:is_table, :char_count])
    
    # Asegurar tipos correctos
    df.is_table = convert(Vector{Bool}, df.is_table)
    df.char_count = convert(Vector{Int}, df.char_count)
    
    println("✅ Cargadas $(nrow(df)) filas válidas")
    
    return df
end

function plot_table_distribution(df, output_file="ocr_table_distribution.png")
    # Contar cuántas son tablas vs no tablas
    table_counts = combine(groupby(df, :is_table), nrow => :count)
    
    labels = ["Non é táboa", "É táboa"]
    counts = [0, 0]
    
    for row in eachrow(table_counts)
        if row.is_table
            counts[2] = row.count
        else
            counts[1] = row.count
        end
    end
    
    total = sum(counts)
    percentages = round.((counts ./ total) .* 100, digits=1)
    
    # Crear etiquetas con números y porcentajes
    labels_with_info = [
        "$(labels[1])\n$(counts[1]) ($(percentages[1])%)",
        "$(labels[2])\n$(counts[2]) ($(percentages[2])%)"
    ]
    
    # Crear pie chart en 3D
    p = pie(labels_with_info, counts,
            title="Distribución de Imaxes: Táboas vs Non-Táboas",
            color=[COLOR_SECONDARY, COLOR_PRIMARY],
            legend=:outertopright,
            size=(1000, 700),
            l=0.5,  # Grosor del borde
            aspect_ratio=:equal,
            camera=(30, 30))  # Ángulo 3D
    
    # Añadir efecto 3D manualmente con sombreado
    # (Plots.jl no soporta pie charts 3D nativamente, pero podemos simular el efecto)
    
    # Guardar
    output_dir = joinpath("memoria", "imaxes")
    mkpath(output_dir)
    output_path = joinpath(output_dir, output_file)
    savefig(p, output_path)
    
    println("\n📊 Distribución de imaxes:")
    println("   Táboas: $(counts[2]) ($(percentages[2])%)")
    println("   Non táboas: $(counts[1]) ($(percentages[1])%)")
    println("   Total: $total")
end

function plot_char_histograms(df, output_file="ocr_char_histogram.png")
    # Filtrar solo las que tienen caracteres (char_count > 0)
    df_with_chars = filter(row -> row.char_count > 0, df)
    
    if nrow(df_with_chars) == 0
        println("⚠️  Non hai imaxes con texto detectado")
        return
    end
    
    # Separar tablas y no tablas
    df_tables = filter(row -> row.is_table, df_with_chars)
    df_non_tables = filter(row -> !row.is_table, df_with_chars)
    
    println("\n📊 Imaxes con texto:")
    println("   Non-táboas: $(nrow(df_non_tables))")
    println("   Táboas: $(nrow(df_tables))")
    
    # Crear plots
    plots_list = []
    
    if nrow(df_non_tables) > 0
        mean_val = mean(df_non_tables.char_count)
        median_val = median(df_non_tables.char_count)
        n = nrow(df_non_tables)
        
        p1 = histogram(df_non_tables.char_count,
                       bins=50,
                       xlabel="Número de caracteres",
                       ylabel="Frecuencia",
                       title="Non-Táboas: Distribución de caracteres (n=$n)",
                       color=COLOR_SECONDARY,
                       legend=:topright,
                       label="",
                       size=(900, 400),
                       alpha=0.7)
        
        # Añadir líneas de media y mediana
        vline!([mean_val], color=:red, linewidth=2, 
               linestyle=:dash, label="Media: $(round(mean_val, digits=1))")
        vline!([median_val], color=:blue, linewidth=2, 
               linestyle=:dot, label="Mediana: $(round(median_val, digits=1))")
        
        push!(plots_list, p1)
    end
    
    if nrow(df_tables) > 0
        mean_val = mean(df_tables.char_count)
        median_val = median(df_tables.char_count)
        n = nrow(df_tables)
        
        p2 = histogram(df_tables.char_count,
                       bins=50,
                       xlabel="Número de caracteres",
                       ylabel="Frecuencia",
                       title="Táboas: Distribución de caracteres (n=$n)",
                       color=COLOR_PRIMARY,
                       legend=:topright,
                       label="",
                       size=(900, 400),
                       alpha=0.7)
        
        # Añadir líneas de media y mediana
        vline!([mean_val], color=:red, linewidth=2, 
               linestyle=:dash, label="Media: $(round(mean_val, digits=1))")
        vline!([median_val], color=:blue, linewidth=2, 
               linestyle=:dot, label="Mediana: $(round(median_val, digits=1))")
        
        push!(plots_list, p2)
    end
    
    if length(plots_list) == 0
        println("⚠️  Non hai datos para crear histogramas")
        return
    end
    
    # Combinar plots
    if length(plots_list) == 2
        p = plot(plots_list..., layout=(2, 1), size=(1000, 900))
    else
        p = plots_list[1]
    end
    
    # Guardar
    output_dir = joinpath("memoria", "imaxes")
    mkpath(output_dir)
    output_path = joinpath(output_dir, output_file)
    savefig(p, output_path)
    
    # Estadísticas
    println("\n📈 Estatísticas de caracteres:")
    
    if nrow(df_non_tables) > 0
        println("\nNon-Táboas (con texto):")
        println("   Total: $(nrow(df_non_tables))")
        println("   Media: $(round(mean(df_non_tables.char_count), digits=1))")
        println("   Mediana: $(median(df_non_tables.char_count))")
        println("   Máximo: $(maximum(df_non_tables.char_count))")
    end
    
    if nrow(df_tables) > 0
        println("\nTáboas (con texto):")
        println("   Total: $(nrow(df_tables))")
        println("   Media: $(round(mean(df_tables.char_count), digits=1))")
        println("   Mediana: $(median(df_tables.char_count))")
        println("   Máximo: $(maximum(df_tables.char_count))")
    end
end

function main()
    csv_path = "crawl/ocr_stats.csv"
    
    # Parse argumentos
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--csv" && i < length(ARGS)
            csv_path = ARGS[i+1]
            i += 2
        else
            i += 1
        end
    end
    
    try
        println("📂 Cargando datos desde: $csv_path")
        df = load_ocr_data(csv_path)
        
        println("\n🎨 Generando gráficos...")
        plot_table_distribution(df, "ocr_table_distribution.png")
        plot_char_histograms(df, "ocr_char_histogram.png")
        
        println("\n✅ Gráficos gardados en memoria/imaxes/")
        return 0
    catch e
        println("❌ ERROR: $e")
        println(stacktrace(catch_backtrace()))
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch e
        println("ERROR: $e")
        exit(1)
    end
end