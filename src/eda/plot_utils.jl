using Plots
using Colors

const COLOR_PRIMARY = RGB(180/255, 59/255, 134/255)
const COLOR_SECONDARY = RGB(179/255, 179/255, 179/255)
const COLOR_ACCENT1 = RGB(60/255, 135/255, 181/255)
const COLOR_ACCENT2 = RGB(135/255, 181/255, 60/255)
const COLOR_ACCENT3 = RGB(181/255, 135/255, 60/255)
const COLOR_ACCENT4 = RGB(135/255, 60/255, 181/255)
const COLOR_ACCENT5 = RGB(50/255, 168/255, 145/255)

const PALETTE = [COLOR_PRIMARY, COLOR_ACCENT1, COLOR_ACCENT2, COLOR_ACCENT3, COLOR_ACCENT4, COLOR_ACCENT5]

function configure_plot_defaults()
    default(
        fontfamily="sans-serif",
        titlefontsize=14,
        guidefontsize=11,
        tickfontsize=9,
        legendfontsize=9,
        background_color=:white,
        foreground_color_subplot=:black,
        grid=true,
        gridalpha=0.15,
        gridstyle=:dot,
        framestyle=:box
    )
end

function save_plot(p, filename; dpi=300)
    output_dir = joinpath("memoria", "imaxes")
    mkpath(output_dir)
    output_path = joinpath(output_dir, filename)
    savefig(p, output_path)
end

function add_value_labels!(p, x, y; offset=0.02, fontsize=9, color=:black)
    y_range = maximum(y) - minimum(y)
    for (xi, yi) in zip(x, y)
        annotate!(p, xi, yi + y_range * offset, text(string(round(Int, yi)), fontsize, color))
    end
end

configure_plot_defaults()

export configure_plot_defaults, save_plot, add_value_labels!
export COLOR_PRIMARY, COLOR_SECONDARY, PALETTE
