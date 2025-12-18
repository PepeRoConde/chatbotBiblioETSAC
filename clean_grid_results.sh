#!/bin/bash
# clean_grid_results.sh
# Clean grid search results while preserving vectorstores

# Default paths
OUTPUT_DIR="memoria/imaxes"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--output-dir DIR]"
            echo "Cleans grid search results while preserving vectorstores."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Directory does not exist: $OUTPUT_DIR"
    exit 1
fi

echo "Scanning: $OUTPUT_DIR"

# Find and delete grid search result files
find "$OUTPUT_DIR" -maxdepth 1 -type f \( \
    -name "grid_search_results.csv" \
    -o -name "grid_search_heatmaps.png" \
    -o -name "grid_search_*.csv" \
    -o -name "grid_search_*.png" \
    -o -name "*grid*.csv" \
    -o -name "*grid*.png" \
\) -print -delete

# Preserve vectorstores (any directory starting with vectorstore)
echo "Vectorstores preserved (if exist):"
find "$OUTPUT_DIR" -maxdepth 1 -type d -name "vectorstore_*" 2>/dev/null

echo "Done."
