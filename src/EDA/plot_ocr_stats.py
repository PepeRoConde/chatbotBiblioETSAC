import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ocr_statistics(csv_path: str = "crawl/ocr_stats.csv"):
    """Generate plots from OCR statistics CSV"""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter out images with 0 chars (failed OCR)
    df_valid = df[df['char_count'] > 0].copy()
    
    print(f"Total images processed: {len(df)}")
    print(f"Valid OCR results: {len(df_valid)}")
    print(f"Tables found: {df_valid['is_table'].sum()}")
    print(f"Non-tables: {(~df_valid['is_table']).sum()}")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== PIE CHART: Table vs Non-Table =====
    table_counts = df_valid['is_table'].value_counts()
    labels = ['Non-Table', 'Table']
    colors = ['#ff9999', '#b43b86']
    
    ax1.pie(table_counts.values, labels=labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title('Images: Tables vs Non-Tables', fontsize=14, fontweight='bold')
    
    # ===== LINE PLOT: Character Count Distribution (Non-Tables only) =====
    non_tables = df_valid[~df_valid['is_table']]['char_count'].sort_values()
    
    if len(non_tables) > 0:
        # Create frequency data
        char_counts = non_tables.values
        frequencies = list(range(1, len(char_counts) + 1))
        
        # Plot
        sns.lineplot(x=char_counts, y=frequencies, ax=ax2, marker='o', markersize=4)
        ax2.set_xlabel('Character Count', fontsize=12)
        ax2.set_ylabel('Frequency (Cumulative)', fontsize=12)
        ax2.set_title('Non-Table Images: Character Count Distribution', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Min: {char_counts.min()}\nMax: {char_counts.max()}\nMean: {char_counts.mean():.1f}"
        ax2.text(0.95, 0.05, stats_text, transform=ax2.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No non-table images found', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    output_path = "memoria/imaxes/ocr.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "crawl/ocr_stats.csv"
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    plot_ocr_statistics(csv_path)
