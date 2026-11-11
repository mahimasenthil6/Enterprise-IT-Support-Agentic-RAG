from app.services.ingestion import load_file,chunk_documents
from pathlib import Path
from app.rag.vectorstore import add_documents
docs=load_file(Path("data/sample_kb/company_it_handbook.md"))
chunks = chunk_documents(docs)
print("Number of chunks:", len(chunks))

result = add_documents(chunks)
print("Stored document IDs:", result)