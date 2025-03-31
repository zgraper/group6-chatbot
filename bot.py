from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

from dotenv import load_dotenv
load_dotenv()


class ChatBot():

    loader = TextLoader('context.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # pinecone.init(
    #     api_key= os.getenv('PINECONE_API_KEY'),
    #     environment='us-east-1-aws'
    # )
    
    
    # index_name = "agriculture-chatbot"
    
   
    # if index_name not in pinecone.list_indexes():
     
    #   pinecone.create_index(name=index_name, metric="cosine", dimension=512)
    #   docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    # else:
     
    #   docsearch = Pinecone.from_existing_index(index_name, embeddings)
        
    # Use FAISS to create a local vector store from the context.txt file
    docsearch = FAISS.from_documents(docs, embeddings)
    
    llm = HuggingFaceHub(
      repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
      model_kwargs={"temperature": 0.8, "max_new_tokens": 150}, 
      huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )


    template = """
    The context is about a agriculture. The human will ask you something about agriculture.
    Use the following piece of context to answer the question in English.
    If you don't know the answer, just say you don't know. 
    
    Context: {context}
    Question: {question}
    Answer: 
    
    """
    
    prompt = PromptTemplate(
      template=template, 
      input_variables=["context", "question"]
    )

    rag_chain = (
      {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
      | prompt 
      | llm
      | StrOutputParser() 
    )