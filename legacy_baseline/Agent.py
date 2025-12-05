import rag_base as rb

if __name__ == "__main__":
    agent = rb.RAGAgent(description="测试知识库")
    with open("test.txt", "r", encoding="utf-8") as f:
        text = f.read()
    agent.add_document(text,"Test Document","2025-09-01")
    res = agent.query("阿响经历了什么故事，学到了什么？")
    print("Query Result:", res)
    
