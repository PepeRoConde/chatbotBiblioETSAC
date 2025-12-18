using JSON3
using Graphs
using GraphPlot
using Colors
using Compose
import Cairo

include("plot_utils.jl")

const COLORS_GRAPH = Dict(
    "pdf" => RGB(179/255, 179/255, 179/255),
    "html" => COLOR_PRIMARY,
    "image" => RGB(0, 255/255, 255/255),
    "root" => RGB(255/255, 107/255, 107/255)
)

function load_metadata(state_dir="crawl")
    metadata_path = joinpath(state_dir, "metadata.json")
    
    if !isfile(metadata_path)
        error("metadata.json non atopado en $state_dir")
    end
    
    metadata = JSON3.read(read(metadata_path, String))
    return metadata
end

function categorize_file(url, file_metadata)
    if haskey(file_metadata, :original_format)
        format_type = lowercase(string(file_metadata.original_format))
        
        if format_type == "pdf"
            return "pdf"
        elseif format_type in ["jpg", "jpeg", "png", "gif", "webp", "svg", "image"]
            return "image"
        elseif format_type in ["html", "htm"]
            return "html"
        end
    end
    
    url_lower = lowercase(url)
    if occursin(".pdf", url_lower)
        return "pdf"
    elseif any(ext -> occursin(ext, url_lower), [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"])
        return "image"
    end
    
    return "html"
end

function build_graph_from_metadata(metadata, base_url=nothing)
    g = DiGraph()
    url_to_idx = Dict{String, Int}()
    idx_to_url = Dict{Int, String}()
    node_types = Dict{Int, String}()
    
    idx = 1
    for (url, data) in pairs(metadata)
        url_str = string(url)
        url_to_idx[url_str] = idx
        idx_to_url[idx] = url_str
        node_type = categorize_file(url_str, data)
        node_types[idx] = node_type
        add_vertex!(g)
        idx += 1
    end
    
    for (url, data) in pairs(metadata)
        url_str = string(url)
        if haskey(data, :discovered_from) && !isnothing(data.discovered_from)
            parent_url = string(data.discovered_from)
            
            if haskey(url_to_idx, parent_url)
                add_edge!(g, url_to_idx[parent_url], url_to_idx[url_str])
            end
        end
    end
    
    if !isnothing(base_url) && haskey(url_to_idx, base_url)
        node_types[url_to_idx[base_url]] = "root"
    end
    
    return g, url_to_idx, idx_to_url, node_types
end

function shorten_url(url, max_len=50)
    if isnothing(url)
        return "None"
    end
    
    # Simplemente truncar de forma segura
    if textwidth(url) <= max_len
        return url
    end
    
    truncated = ""
    for c in url
        if textwidth(truncated * c) > max_len - 3
            return truncated * "..."
        end
        truncated *= c
    end
    
    return truncated
end


function visualize_graph(g, node_types, idx_to_url, output_file="crawler_graph.png"; 
                        layout_type="spring", silent=false)
    if nv(g) == 0
        return
    end
    
    node_colors = [COLORS_GRAPH[node_types[i]] for i in 1:nv(g)]
    node_sizes = [node_types[i] == "root" ? 0.08 : 0.04 for i in 1:nv(g)]
    
    if layout_type == "spring"
        layout = spring_layout
    elseif layout_type == "circular"
        layout = circular_layout
    else
        layout = spring_layout
    end
    
    labels = silent ? nothing : [shorten_url(idx_to_url[i], 40) for i in 1:nv(g)]
    
    p = gplot(g, 
             layout=layout,
             nodefillc=node_colors,
             nodesize=node_sizes,
             nodelabel=labels,
             nodelabeldist=2.0,
             nodelabelsize=1.5,
             edgestrokec=colorant"gray",
             arrowlengthfrac=0.02)
    
    output_dir = joinpath("memoria", "imaxes")
    mkpath(output_dir)
    output_path = joinpath(output_dir, output_file)
    
    draw(PNG(output_path, 20inch, 16inch), p)
end

function main()
    state_dir = "crawl"
    output_file = "grafo.png"
    layout_type = "spring"
    base_url = nothing
    silent = false
    
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--state-dir" && i < length(ARGS)
            state_dir = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--output" && i < length(ARGS)
            output_file = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--layout" && i < length(ARGS)
            layout_type = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--base-url" && i < length(ARGS)
            base_url = ARGS[i+1]
            i += 2
        elseif ARGS[i] in ["-s", "--silent"]
            silent = true
            i += 1
        else
            i += 1
        end
    end
    
    state_path = abspath(state_dir)
    
    if !isdir(state_path)
        println("ERROR: State directory not found: $state_path")
        return 1
    end
    
    try
        metadata = load_metadata(state_path)
        g, url_to_idx, idx_to_url, node_types = build_graph_from_metadata(metadata, base_url)
        visualize_graph(g, node_types, idx_to_url, output_file, layout_type=layout_type, silent=silent)
        return 0
    catch e
        println("ERROR in visualize_graph main: $e")
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch e
        println("ERROR in visualize_graph: $e")
        exit(1)
    end
end
