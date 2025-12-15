import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_from_faiss():
    """Construye índice TF-IDF usando los mismos chunks de FAISS"""
    
    # 1. Cargar los chunks de FAISS
    print("Cargando chunks de index.pkl...")
    with open('local_vectorstore/index.pkl', 'rb') as f:
        data = pickle.load(f)
    
    docstore = data[0]
    index_to_id = data[1]
    
    # 2. Extraer textos de los chunks (en el mismo orden que FAISS)
    chunk_texts = []
    chunk_ids = []
    
    for idx in sorted(index_to_id.keys()):
        doc_id = index_to_id[idx]
        doc = docstore._dict[doc_id]
        chunk_texts.append(doc.page_content)
        chunk_ids.append(doc_id)
    
    print(f"Cargados {len(chunk_texts)} chunks")
    
    # 3. Construir TF-IDF
    print("Construyendo índice TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)
    
    print(f"Matriz TF-IDF: {tfidf_matrix.shape}")
    
    # 4. Guardar
    tfidf_data = {
        'vectorizer': vectorizer,
        'matrix': tfidf_matrix,
        'chunk_ids': chunk_ids,
        'index_to_id': index_to_id  # Para mantener correspondencia con FAISS
    }
    
    with open('tfidf_index.pkl', 'wb') as f:
        pickle.dump(tfidf_data, f)
    
    print("✓ Índice TF-IDF guardado en tfidf_index.pkl")
    return tfidf_data

if __name__ == "__main__":
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    build_tfidf_from_faiss()