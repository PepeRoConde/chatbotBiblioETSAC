import pickle
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search_tfidf(query, top_k=5):
    """Busca en el índice TF-IDF"""
    
    # 1. Cargar índice TF-IDF
    print("Cargando índice TF-IDF...")
    with open('tfidf_index.pkl', 'rb') as f:
        tfidf_data = pickle.load(f)
    
    vectorizer = tfidf_data['vectorizer']
    tfidf_matrix = tfidf_data['matrix']
    chunk_ids = tfidf_data['chunk_ids']
    
    # 2. Cargar docstore para obtener los textos completos
    with open('local_vectorstore/index.pkl', 'rb') as f:
        data = pickle.load(f)
    docstore = data[0]
    
    # 3. Vectorizar la query
    query_vec = vectorizer.transform([query])
    
    # 4. Calcular similitudes
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 5. Obtener top_k resultados
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 6. Mostrar resultados
    print(f"\n Query: '{query}'\n")
    print(f"Top {top_k} resultados:\n")
    
    for rank, idx in enumerate(top_indices, 1):
        doc_id = chunk_ids[idx]
        doc = docstore._dict[doc_id]
        score = similarities[idx]
        
        print(f"{rank}. Score: {score:.4f}")
        print(f"   {doc.page_content}")
        print()
    
    return top_indices, similarities

if __name__ == "__main__":
    # Prueba con una búsqueda
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    search_tfidf("préstamos biblioteca", top_k=10)