#!/bin/bash

# Use first parameter if provided, otherwise default to 'crawl'
DIR="${1:-crawl}"

echo "Cleaning directory: $DIR"

# Remove files and directories
rm -rf "$DIR/visited_urls.txt"
rm -rf "$DIR/metadata.json"
rm -rf "$DIR/map.json"
rm -rf "$DIR/crawled_data"
rm -rf "$DIR/text"

echo "Done! Cleaned $DIR directory."
