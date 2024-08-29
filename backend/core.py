import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = None):
    if chat_history is None:
        chat_history = []
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    docsearch = PineconeVectorStore(index_name=os.environ['PINECONE_INDEX'], embedding=embeddings)
    chat = ChatOllama(model='mistral', temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == '__main__':
    res = run_llm(query="What is a LangChain Chain?")
    print(res["result"])
