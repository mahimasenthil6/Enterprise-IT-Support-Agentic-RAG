from app.services.ingestion import load_file,chunk_documents
from pathlib import Path
docs=load_file(Path("data/sample_kb/company_it_handbook.md"))
docs= chunk_documents(docs)
print(len(docs))