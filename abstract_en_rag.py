from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import json
import logging
from typing import List, Dict
import re
import openai

app = Flask(__name__)


client = openai.OpenAI()


bad_words = [
    # General Offensive Language
    "fuck", "fucking", "shit", "asshole", "bitch", "bastard", "dick", "pussy", 
    "cunt", "cock", "dildo", "slut", "whore", "crap", "piss", "damn", "hell", 
    "jerk", "twat", "bollocks", "tosser", "wanker",
    # Racial Slurs
    "nigger", "chink", "gook", "spic", "wetback", "beaner", "cracker", "honky", 
    "paki", "gypsy", "coon", "jigaboo", "sandnigger", "kike",
    # Sexual Orientation and Gender Identity Slurs
    "faggot", "dyke", "tranny", "lesbo", "queer", "homo", "he-she", "shemale", "fudgepacker",
    # Religious Offense
    "infidel", "heathen", "blasphemy", "heretic", "jihadist", "terrorist", "zionist", 
    "crusader", "taliban", "kafir",
    # Derogatory Terms for Disabilities
    "retard", "cripple", "lame", "dumb", "spaz", "idiot", "moron", "lunatic",
    # Offensive Terms Regarding Appearance
    "fat", "ugly", "lardass", "tubby", "beanpole", "midget", "dwarf", "four-eyes",
    # Violent and Threatening Terms
    "kill", "die", "murder", "rape", "shoot", "stab", "bomb", "suicide", "lynch", 
    "execute", "terrorist", "massacre",
    # Offensive Terms About Mental Health
    "psycho", "nutjob", "crazy", "schizo", "maniac",
    # Miscellaneous Offensive Terms
    "garbage", "trash", "scum", "filth", "vermin", "loser", "scumbag", "degenerate", 
    "dirtbag", "slimeball", "harlot", "low-life"
]

bad_words_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in bad_words) + r')\b', re.IGNORECASE)

def check_for_bad_words(user_input):
    if bad_words_pattern.search(user_input):
        return "Input blocked: Inappropriate language detected!"
    return None

def censor_bad_words(user_input):
    return bad_words_pattern.sub(lambda match: '*' * len(match.group()), user_input)
# creating embedings of corpus
class Ingestion:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    def load_docs(self, file_paths: List[str]) -> List[Dict]:
        docs = []
        for file in file_paths:
            loader = PyPDFLoader(file)
            docs.extend(loader.load())
        return docs

    def create_chunks(self, docs: List[Dict]) -> List[Dict]:
        return self.text_splitter.split_documents(docs)

    def create_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        return self.embedding.embed_documents([chunk.page_content for chunk in chunks])
# storing embeds 
class Storage:
    def __init__(self):
        self.vectorstore = None

    def store_embeddings(self, chunks: List[Dict], embeddings: List[List[float]]):
        self.vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        return self.vectorstore.similarity_search(query, k=k)
# the response 
class Generator:
    def __init__(self, client, storage: Storage):
        self.client = client
        self.storage = storage

    def generate_response(self, user_input: str, context: str):
        messages = [
            {"role": "system", "content": """You are a helpful assistant. Answer the question using the provided context from corporate documents. 
            Along with your answer, provide a confidence score between 0 and 1, where 0 means you're not at all confident and 1 means you're absolutely certain.
            Format your response as JSON with 'answer' and 'confidence' fields.
            If you cannot find a relevant answer based on the given context, set the confidence to 0.
            Ensure that you filter out any inappropriate or offensive language in your response."""},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,
        )
        return json.loads(response.choices[0].message.content)

    def ask_question(self, user_input: str) -> Dict:
        try:
            bad_language_message = check_for_bad_words(user_input)
            if bad_language_message:
                return {"error": bad_language_message}, 400

            censored_input = censor_bad_words(user_input)
            docs = self.storage.similarity_search(censored_input)

            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                result = self.generate_response(censored_input, context)
                
                if result['confidence'] > 0.5:  # You can adjust this threshold
                    response_text = f"Answer (confidence: {result['confidence']}): {result['answer']}"
                else:
                    response_text = "I don't have enough confident information to answer this question."
            else:
                response_text = "No relevant information found to answer the question."

            return {"response": response_text}
        except Exception as e:
            logging.error(f"Error in ask_question: {str(e)}")
            return {"error": "An error occurred while processing your request"}, 500

# Initialize classes
ingestion = Ingestion()
storage = Storage()
generator = Generator(client, storage)

# Load and process documents -  done during app initialization)
file_paths = [
    "SIH/prd.pdf",
    "SIH/playbook.pdf",
    "SIH/it_support.PDF",
    "SIH/hr.pdf",
    "SIH/corp_events.pdf",
]
documents = ingestion.load_docs(file_paths)
chunks = ingestion.create_chunks(documents)
embeddings = ingestion.create_embeddings(chunks)
storage.store_embeddings(chunks, embeddings)

@app.route("/ask", methods=["POST"])
def ask_question_endpoint():
    user_input = request.json.get("question")
    result = generator.ask_question(user_input)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
