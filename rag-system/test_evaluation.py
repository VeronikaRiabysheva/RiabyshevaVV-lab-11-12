from retriever.vector_store import VectorStore
from evaluation.test_dataset import EvaluationDataset
from evaluation.retrieval_evaluator import RetrievalEvaluator
from documents.tech_docs import DOCUMENTS

vector_store = VectorStore(collection_name="rag_documents") 
if vector_store.get_collection_info()["document_count"] == 0:
    print("Adding documents to ChromaDB...")
    vector_store.add_documents(DOCUMENTS)
else:
    print("Documents already in ChromaDB.")

dataset = EvaluationDataset()
evaluator = RetrievalEvaluator(vector_store)

results = evaluator.evaluate_dataset(dataset.test_cases)
print(results)