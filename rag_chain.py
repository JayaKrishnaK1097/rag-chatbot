from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def build_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {input}

Answer:"""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    return chain

if __name__ == "__main__":
    chain = build_chain()
    result = chain.invoke({"input": "What is the document about?"})
    print(result["answer"])