import asyncio
import sqlite3
from datasets import Dataset
from ragas import evaluate, dataset_schema
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from rag_base import RAGAgent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm.asyncio import tqdm
from typing import cast

import argparse

async def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system using Ragas')
    parser.add_argument('--limit', type=int, default=10, help='Number of questions to evaluate (default: 10)')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrency for RAG queries (default: 5)')
    args = parser.parse_args()

    # Initialize RAG Agent
    agent = RAGAgent()
    
    # Connect to DB and fetch benchmark data
    conn = sqlite3.connect('./sqlite_db/sqlite.db')
    cursor = conn.cursor()
    
    # Check if benchmark_qa table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='benchmark_qa'")
    if not cursor.fetchone():
        print("Error: benchmark_qa table not found. Please generate benchmark dataset first.")
        return

    cursor.execute("SELECT question, answer FROM benchmark_qa")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No benchmark data found.")
        return
        
    # Limit for testing purposes
    if args.limit > 0 and len(rows) > args.limit:
        print(f"Found {len(rows)} benchmark questions. Limiting to {args.limit} for testing.")
        rows = rows[:args.limit]
    else:
        print(f"Found {len(rows)} benchmark questions. Starting evaluation...")
    
    questions = [row[0] for row in rows]
    # Ragas 0.3+ SingleTurnSample expects 'reference' (mapped from ground_truth) to be a string, not a list
    ground_truths = [row[1] for row in rows] 
    
    answers = []
    contexts = []
    
    # Run RAG asynchronously
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def process_question(q):
        async with semaphore:
            try:
                result = await agent.async_query(q)
                return result
            except Exception as e:
                print(f"Error processing question '{q}': {e}")
                return None

    tasks = [process_question(q) for q in questions]
    results = await tqdm.gather(*tasks, desc="Processing RAG queries")
    
    valid_indices = []
    for i, res in enumerate(results):
        if res:
            answers.append(res["answer"])
            # Extract content from retrieved docs
            ctx = [doc["content"] for doc in res["sources"]]
            contexts.append(ctx)
            valid_indices.append(i)
    
    # Filter questions and ground_truths to match successful results
    questions = [questions[i] for i in valid_indices]
    ground_truths = [ground_truths[i] for i in valid_indices]
    
    if not questions:
        print("No successful RAG queries.")
        return

    # Prepare dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Running Ragas evaluation...")
    
    # Configure Ragas to use the same LLM/Embeddings as the Agent
    # We need to wrap them in LangChain objects
    
    if not agent.llm_model or not agent.embedding_model:
        raise ValueError("Agent configuration incomplete")

    llm = ChatOpenAI(
        model=agent.llm_model,
        api_key=agent.llm_api_key, # type: ignore
        base_url=agent.llm_base_url
    )
    
    embeddings = OpenAIEmbeddings(
        model=agent.embedding_model,
        api_key=agent.embedding_api_key, # type: ignore
        base_url=agent.embedding_base_url
    )
    
    # Run evaluation
    results = cast(dataset_schema.EvaluationResult, evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings
    ))
    
    print("\nEvaluation Results:")
    print(results)
    
    # Save results
    df = results.to_pandas()
    df.to_csv("ragas_evaluation_results.csv", index=False)
    print("Results saved to ragas_evaluation_results.csv")

if __name__ == "__main__":
    asyncio.run(main())
