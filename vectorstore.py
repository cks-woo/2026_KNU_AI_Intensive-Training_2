import csv
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from embeddings import get_embeddings
from typing import List

def load_documents() -> List[Document]:
    docs = []
    # dataset.csv 파일을 읽어 Document 객체로 변환
    with open('dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # LLM이 ID와 내용을 명확히 인지하도록 구성
            content = f"규정 ID: {row['id']}\n분류: {row['category']} - {row['sub_category']}\n제목: {row['title']}\n내용: {row['content']}"
            
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'id': row['id'],
                        'category': row['category'],
                        'sub_category': row['sub_category'],
                        'title': row['title']
                    }
                )
            )
    return docs

def embedding(docs: List[Document]):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=docs, # CSV 텍스트 길이가 짧으므로 청킹(split_docs) 생략
        embedding=embeddings
    )
    return vectorstore

def save_vector_to_local(vectorstore):
    path_str = './exp-faiss'    
    vectorstore.save_local(path_str)

def load_vector_from_local():
    path_str = './exp-faiss'
    return FAISS.load_local(
        path_str,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def init_vectorstore():
    docs = load_documents()
    vectorstore = embedding(docs)
    save_vector_to_local(vectorstore)
    return vectorstore