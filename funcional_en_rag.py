import json
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import logging
from typing import List, Dict

app = Flask(__name__)

# Assuming these are defined elsewhere
client = None  # OpenAI client
image_collection = None  # MongoDB collection for images
check_for_bad_words = None  # Function to check for bad words

def ingest_documents(file_paths: List[str]) -> List[Dict]:
    """Ingest documents and create chunks."""
    pages = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_documents(pages)
    return chunks

def create_vectorstore(chunks: List[Dict]) -> Chroma:
    """Create and return a vector store from document chunks."""
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding)
    return vectorstore

def generate_response(user_input: str, context: str, model: str = "gpt-4") -> Dict:
    """Generate a response using the OpenAI API."""
    messages = [
        {"role": "system", "content": """You are a helpful assistant. Answer the question using the provided context from corporate documents. 
        Along with your answer, provide a confidence score between 0 and 1, where 0 means you're not at all confident and 1 means you're absolutely certain.
        Format your response as JSON with 'answer' and 'confidence' fields.
        If you cannot find a relevant answer based on the given context, set the confidence to 0.
        Ensure that you filter out any inappropriate or offensive language in your response."""},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
    )
    return json.loads(response.choices[0].message.content)

def search_corporate_data(vectorstore: Chroma, user_input: str) -> Tuple[bool, str]:
    """Search corporate data and generate a response."""
    docs = vectorstore.similarity_search(user_input, k=3)
    if docs:
        context = "\n".join([doc.page_content for doc in docs])
        result = generate_response(user_input, context)
        if result['confidence'] > 0.5:
            return True, f"Answer from corp data (confidence: {result['confidence']}): {result['answer']}"
    return False, ""

def search_image_analysis(user_input: str) -> Tuple[bool, str]:
    """Search image analysis data and generate a response."""
    relevant_images = list(image_collection.find().sort("_id", -1).limit(5))
    if relevant_images:
        image_descriptions = [img['analysis'] for img in relevant_images]
        combined_context = "\n".join(image_descriptions)
        result = generate_response(user_input, combined_context)
        if result['confidence'] > 0.5:
            return True, f"Answer from image analysis (confidence: {result['confidence']}): {result['answer']}"
    return False, ""

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        user_input = request.json.get("question")
        bad_language_message = check_for_bad_words(user_input)
        
        if bad_language_message:
            return jsonify({"error": bad_language_message}), 400

        # Ingest and store documents (this could be done at app startup and stored)
        file_paths = [
            "SIH/prd.pdf",
            "SIH/playbook.pdf",
            "SIH/it_support.PDF",
            "SIH/hr.pdf",
            "SIH/corp_events.pdf",
        ]
        chunks = ingest_documents(file_paths)
        vectorstore = create_vectorstore(chunks)

        # Search corporate data
        answer_found, response_text = search_corporate_data(vectorstore, user_input)

        # If no answer from corporate data, try image analysis
        if not answer_found:
            answer_found, response_text = search_image_analysis(user_input)

        # If still no answer, return default message
        if not answer_found:
            response_text = "I don't have enough confident information to answer this question."

        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Error in ask_question endpoint: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500

if __name__ == "__main__":
    app.run(debug=True)
