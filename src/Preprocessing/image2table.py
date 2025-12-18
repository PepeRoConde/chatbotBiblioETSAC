import os
import sys
import subprocess
from img2table.document import Image
from img2table.ocr import TesseractOCR

def image_table2text(IMAGE_PATH):
    """Extract table data from image with suppressed tesseract output.
    
    Args:
        IMAGE_PATH: Path to image file (str or Path object)
        
    Returns:
        str: Extracted table text or error message
    """
    # Convert Path object to string if needed
    image_path_str = str(IMAGE_PATH)
    
    # Suppress stderr (where tesseract prints its version info)
    old_stderr = sys.stderr
    
    try:
        sys.stderr = open(os.devnull, 'w')
        
        ocr = TesseractOCR(lang="eng")
        doc = Image(image_path_str)
        tables = doc.extract_tables(ocr=ocr)
        
    except Exception as e:
        return f"Error processing image: {e}"
    finally:
        # Always restore stderr
        sys.stderr.close()
        sys.stderr = old_stderr
    
    text = '' 
    
    for i, table in enumerate(tables):
        text += f"\n=== Taboa {i+1} ===\n"
        
        df = table.df
        
        if len(df) > 0:
            headers = df.iloc[0].values
            
            # For subsequent rows, prepend column name to each cell
            for idx in range(1, len(df)):
                row_values = []
                for col_idx, cell_value in enumerate(df.iloc[idx].values):
                    col_name = str(headers[col_idx])
                    # Replace newlines and multiple spaces with single space
                    clean_value = str(cell_value).replace('\n', ' ').replace('\r', ' ')
                    clean_value = ' '.join(clean_value.split())  # Remove extra whitespace
                    prefixed_value = f"[{col_name}]-{clean_value}"
                    row_values.append(prefixed_value)
                # Fixed: proper string concatenation
                text += f'fila {idx}: ' + " ".join(row_values) + '\n'
        else:
            text += 'non é unha taboa (ou esta vacia)\n'
    
    return text if text else 'non se atoparon taboas'


# Example usage
if __name__ == "__main__":
    result = image_table2text("memoria/imaxes/prestamos.png")
    print(result)
