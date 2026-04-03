from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document

SYSTEM_PROMPT = """당신은 KBO 야구 규정 전문 AI 어시스턴트입니다.
반드시 아래 제공된 [참고 문서]의 내용만을 바탕으로 답변하세요.

규칙:
1. 근거 제시: 답변 시 반드시 참고한 규정의 ID를 명시하세요. (예: "ID 4 규정에 따르면...")
2. 논리적 추론: 여러 규정이 관련된 상황이라면(예: ABS, 강우콜드, 서스펜디드가 겹친 상황), 각 규정이 어떻게 적용되어 최종 결론에 도달하는지 단계별로 명확히 서술하세요.
3. 날짜 계산: 소급 적용이나 복귀 날짜를 묻는 경우, 규정된 일수를 엄격히 적용하여 최종 날짜를 명확히 계산해 내세요. 주의사항이나 예외 조건이 있다면 함께 안내하세요.
4. 참고 문서에 없는 내용은 "해당 내용은 규정에서 확인할 수 없습니다."라고 안내하세요.

[참고 문서]
{context}"""

def build_rag_chain(vectorstore):
    load_dotenv()

    llm = ChatGroq(
        model="llama-3.1-8b-instant"
    )

    # 여러 규정을 조합해야 하므로 검색 개수(k)를 3에서 5로 증가
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "관련 문서를 찾지 못했습니다."
    
    sections = []
    for i, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get("id", "N/A")
        title = doc.metadata.get("title", "")
        # LLM이 문맥을 잘 파악하도록 메타데이터를 프롬프트 텍스트로 주입
        section = f"--- 문서 {i} (ID: {doc_id}) ---\n제목: {title}\n내용:\n{doc.page_content}"
        sections.append(section)

    return "\n\n".join(sections)