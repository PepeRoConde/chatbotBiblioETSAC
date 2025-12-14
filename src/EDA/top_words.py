import argparse
import os
from collections import Counter
import re
import matplotlib.pyplot as plt
import csv


def read_text_files(directory='.'):
    """Read all .txt files in the specified directory."""
    text = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename != 'top_words.txt':
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text.append(f.read())
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    return ' '.join(text)


def extract_words(text):
    """Extract words from text, converting to lowercase."""
    # Remove punctuation and split into words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words


def get_top_words(words, n):
    """Get the top n+5 most common words."""
    counter = Counter(words)
    # Get top n+5 words for the plot (n colored, 5 gray for context)
    return counter.most_common(n + 5)


def save_top_words_csv(top_words, n, output_file='top_words.csv'):
    """Save the top n words to a CSV file."""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for word, count in top_words[:n]:
            writer.writerow([word, count])
    print(f"Top {n} words saved to {output_file}")


def load_stopwords_from_csv(csv_file='top_words.csv'):
    """Load words from CSV file to use as stopwords.
    
    Returns:
        List of words (first column of CSV)
    """
    stopwords = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Check if row is not empty
                    stopwords.append(row[0])
    except FileNotFoundError:
        print(f"Warning: {csv_file} not found")
    return stopwords


def create_barplot(top_words, n):
    """Create a bar plot of the top words with custom colors."""
    words = [word for word, count in top_words]
    counts = [count for word, count in top_words]
    
    # Create color list: first n are RGB(180,59,134), rest are gray
    colors = []
    primary_color = (180/255, 59/255, 134/255)  # Convert RGB to matplotlib format
    gray_color = (0.7, 0.7, 0.7)
    
    for i in range(len(words)):
        if i < n:
            colors.append(primary_color)
        else:
            colors.append(gray_color)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(words)), counts, color=colors)
    
    # Customize the plot
    plt.xlabel('Palabras', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title(f'Top {n} Palabras (+ 5 para contexto)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('memoria/imaxes/palabras.png', dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(
        description='Analyze text files and extract the most common words.'
    )
    parser.add_argument(
        '-n', 
        type=int, 
        default=15,
        help='Number of top words to extract (default: 15)'
    )
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='crawl/text/',
        help='Directory containing text files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Read all text files
    print(f"Reading text files from {args.directory}...")
    text = read_text_files(args.directory)
    
    if not text:
        print("Error: No text files found or all files are empty.")
        return
    
    # Extract words
    words = extract_words(text)
    print(f"Total words found: {len(words)}")
    print(f"Unique words: {len(set(words))}")
    
    # Get top words
    top_words = get_top_words(words, args.n)
    
    if len(top_words) < args.n:
        print(f"Warning: Only {len(top_words)} unique words found.")
    
    # Save top n words to CSV file
    save_top_words_csv(top_words, args.n)
    
    # Create bar plot
    create_barplot(top_words, args.n)


if __name__ == '__main__':
    main()
