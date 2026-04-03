from vectorstore import init_vectorstore, load_vector_from_local
from chain import build_rag_chain

# 최초 실행 시 아래 주석을 풀고 vectorstore를 생성하세요.
# init_vectorstore()

vectorstore = load_vector_from_local()
chain = build_rag_chain(vectorstore=vectorstore)

# 요구사항 1
q1 = "피치클락 위반 시 타자와 투수에게 각각 어떤 페널티가 부여되나요?"
print(f"Q: {q1}")
print(chain.invoke(q1))
print("-" * 50)

# 요구사항 2 (ID 8, 9, 10 조합 추론)
q2 = "A선수가 3일 전 경기에 나갔고 오늘 부상으로 말소됐다면, 소급 적용을 포함해 언제 복귀 가능한가?"
print(f"Q: {q2}")
print(chain.invoke(q2))
print("-" * 50)

# 요구사항 3 (ID 2, 24, 23 종합 판단)
q3 = "경기 중 ABS가 고장 났고, 5회초에 비가 내려 경기가 중단된 상황입니다. 현재 경기의 최종 상태(종료/일시중단)는 어떻게 판정되나요?"
print(f"Q: {q3}")
print(chain.invoke(q3))
print("-" * 50)