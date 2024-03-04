# import os
# import sys
# import openai
# from flask import Flask, request, jsonify
# from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader, TextLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain.llms import OpenAI
# from langchain.vectorstores import Chroma

# import constants;

# os.environ["OPENAI_API_KEY"] = constants.APIKEY

# app = Flask(__name__)

# # Define the route for the root URL ("/") and provide a welcome message
# @app.route('/')
# def index():
#     return 'Welcome to Langchain Chatbot'

# # Rest of your code...

# # Enable to save to disk & reuse the model (for repeated queries on the same data)
# PERSIST = False

# query = None

# if PERSIST and os.path.exists("persist"):
#     print("Reusing index...\n")
#     vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
#     index = VectorStoreIndexWrapper(vectorstore=vectorstore)
# else:
#     # loader = TextLoader("data.txt")  # Use this line if you only need data.txt
#     # loader = DirectoryLoader("data/")
#     loader = TextLoader("data/data.txt")

#     if PERSIST:
#         index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
#     else:
#         index = VectorstoreIndexCreator().from_loaders([loader])

# chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(model="gpt-3.5-turbo"),
#     retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
# )

# chat_history = []

# @app.route('/query', methods=['POST'])
# def query_endpoint():
#     data = request.json
#     query = data.get('query')
       
#     if not query:
#         return jsonify({"error": "Missing 'query' parameter"}), 400

#     result = chain({"question": query, "chat_history": chat_history})
#     return jsonify({"answer": result['answer']})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000)




import os
from flask import Flask, request, jsonify, send_from_directory
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.llms import PromptLayerOpenAI
# import os
import sys
import openai
from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

# import constants;
# OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = "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"

app = Flask(__name__)

# Define the route for the root URL ("/") and provide a welcome message
@app.route('/')
def index():
    # Serve the HTML file from the 'static' folder
    return send_from_directory('static', 'index.html')

# Rest of your code...

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    # loader = TextLoader("data.txt")  # Use this line if you only need data.txt
    # loader = DirectoryLoader("data/")
    loader = TextLoader("data/data.txt")

    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.json
    query = data.get('query')
       
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    result = chain({"question": query, "chat_history": chat_history})
    return jsonify({"answer": result['answer']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)



