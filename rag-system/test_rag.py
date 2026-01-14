import asyncio
from pipeline.rag_pipeline import RAGPipeline
from retriever.vector_store import VectorStore
from documents.tech_docs import DOCUMENTS

async def main():
    retriever = VectorStore(collection_name="rag_documents")
    if retriever.get_collection_info()["document_count"] == 0:
        print("Adding documents to ChromaDB...")
        retriever.add_documents(DOCUMENTS)
    else:
        print("Documents already in ChromaDB.")

    # Тест сырого поиска (без пайплайна)
    test_query = "Что такое AI?"
    raw_docs = retriever.search(test_query, n_results=5)
    print(f"Raw search results for '{test_query}':")
    print(raw_docs)  # Здесь увидите документы и их similarity_score

    # Теперь пайплайн
    pipeline = RAGPipeline()
    result = await pipeline.process_question(test_query)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())