import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    st.error(
        "Google API Key is missing! Please add your API Key in the environment variables."
    )
    st.stop()
genai.configure(api_key=api_key)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Enable dangerous deserialization
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question)

    if not docs:
        st.write(
            "No relevant documents found for your question. Let's see what the AI thinks..."
        )
        # Use the AI model directly for a generic answer when no docs are found
        ai_fallback_response = generate_fallback_answer(user_question)
        st.write("AI's Response: ", ai_fallback_response)
        return

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    answer = response.get("output_text", "No response available")

    # Custom fallback if the context doesn't match the user's question
    if "answer is not available in the context" in answer.lower():
        st.write(
            "Let's see what the AI thinks..."
        )
        # Use the AI model directly for a generic answer
        ai_fallback_response = generate_fallback_answer(user_question)
        st.write("AI's Response: ", ai_fallback_response)
    else:
        st.write("Reply: ", answer)


def generate_fallback_answer(user_question):
    """Use Google Generative AI to generate an answer when no context matches the user's question."""
    ai_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    # Properly structure the input as a list of `BaseMessage` instances
    from langchain.schema import HumanMessage

    messages = [HumanMessage(content=user_question)]

    # Call the model with the correct input
    response = ai_model(messages)

    # The response is an AIMessage object, so we can directly access its content
    return response.content


def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":book:")
    st.header("Chat with PDF using GenAI")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file before submitting.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
