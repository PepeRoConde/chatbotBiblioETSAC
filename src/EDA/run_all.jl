const SCRIPTS = [
    ("analyze_last_modified.jl", "last_modified_analysis.png"),
    ("analyze_dates_in_content.jl", "histograma_datas.png"),
    ("plot_ocr_stats.jl", "ocr.png"),
    ("top_words.jl", "palabras.png"),
    ("visualize_graph.jl", "grafo.png"),
    ("zipf_analyzer.jl", "zipf_modelo.png"),
    ("zipf_varios.jl", "zipf_modelo_normalizado.png")
]

function main()
    mkpath("memoria/imaxes")
    
    println("Running EDA plotting scripts...")
    println("="^60)
    
    for (script, expected_output) in SCRIPTS
        if isfile(script)
            print("Running $script... ")
            try
                result = run(`julia $script`)
                if success(result)
                    output_path = joinpath("memoria/imaxes", expected_output)
                    if isfile(output_path)
                        println("✓ OK ($expected_output)")
                    else
                        println("⚠ Script ran but no plot: $expected_output")
                    end
                else
                    println("✗ FAILED")
                end
            catch e
                println("✗ ERROR: $e")
            end
        else
            println("⚠ Script not found: $script")
        end
    end
    
    println("="^60)
    println("Check plots in: memoria/imaxes/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
