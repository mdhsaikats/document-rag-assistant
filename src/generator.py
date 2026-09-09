from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def answer_question(vector_db, question: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    system_prompt = (
        "You are a helpful assistant. Use ONLY the following retrieved context "
        "to answer the question. If you don't know the answer based on the context, "
        "say 'I cannot answer this based on the provided document.'\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    response = rag_chain.invoke({
        "input": question
    })

    return response["answer"]