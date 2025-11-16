from DocumentProcessor import DocumentProcessor
from pathlib import Path

# Supongamos que tienes un documento en crawl/crawled_data
docs_folder = "crawl/crawled_data"

processor = DocumentProcessor(
    docs_folder=docs_folder,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=300,   # tamaño de cada chunk
    chunk_overlap=50, # solapamiento
    verbose=True,
    cache_dir=".doc_cache",
    prefix_mode="source"
)

# Cargar documentos (solo los nuevos o modificados)
processor.load_documents(force_reload=True)

# Dividir documentos en chunks
chunks = processor.split_documents()
print(f"\nNúmero total de chunks generados: {len(chunks)}")

# Filtrar chunks de un documento específico
# doc_name = "04-05-NORMATIVA-PRUEBA-DE-APTITUD-Aprobada-6-03-20_922ec9e5.pdf"
doc_name = "06_Calendario_Avaliac_Compens_25_26_ba5c052a.pdf"
# doc_name = "es_00cd51a6.html"

doc_chunks = [c for c in chunks if Path(c.metadata.get("source_file")).name == doc_name]

print(f"\nNúmero de chunks para el documento {doc_name}: {len(doc_chunks)}\n")

for i, chunk in enumerate(doc_chunks):
    print(f"---- Chunk {i+1} ----")
    print("Fuente:", chunk.metadata.get("source_file"))
    print("Hash:", chunk.metadata.get("file_hash"))
    print("Contenido completo:\n", chunk.page_content)
    print("---------------------------\n")
