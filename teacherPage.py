CHUNK_SIZE = 1096
CHUNK_OVERLAP = 10
FETCH_COUNT = 2
NUM_PROBLEMS = 5
SESSION_ID = "Default"
DIRECTORY = "./"
LANGUAGE = "english"

import os
#__import__('pysqlite3')
import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import requests
#import sqlite3
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_google_firestore import FirestoreVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.firestore import FirestoreChatMessageHistory
from langchain_google_firestore import FirestoreChatMessageHistory
from mymath import *
from studentPage import *
from teacherPage import *
import json
import firebase_admin
from firebase_admin import credentials, firestore
PROJECT_ID = "for-test-4cb36"
os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "for_test_credentials.json"
#set GOOGLE_APPLICATION_CREDENTIALS= "for_test_credentials.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
DB_FILE = "file_database.json"
UPLOAD_DIR = "uploaded_files"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def load_file_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_file_database(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f)


def delete_file(class_name):
    file_db = load_file_database()
    if class_name in file_db:
        file_path = file_db[class_name]['path']
        if os.path.exists(file_path):
            os.remove(file_path)
        del file_db[class_name]
        save_file_database(file_db)
        
        # 删除 Firestore 中的向量存储
        db = firestore.client()
        db.collection(class_name).delete()
        
        return True
    return False
def store_docs(file, class_name):
    documents = []
    filepath= file.name
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())
    file_db = load_file_database()
    file_db[class_name] = {
             "name": file.name,
             "path": filepath
        }
    save_file_database(file_db)
    loader = PyPDFLoader(filepath)
    documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size= CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = text_splitter.split_documents(documents)

    # Convert the document chunks to embedding and save them to the vector store
    vector_store = FirestoreVectorStore.from_documents(
        collection = class_name,
        documents = documents,
        embedding=OpenAIEmbeddings())

# 老師的頁面
def teacherPage():
    st.title('Teacher Page')
    st.write('Welcome, Teacher!')
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    class_name = st.text_input("为文件指定一个唯一代码")

    if uploaded_file is not None and class_name:
        if st.button("上传文件"):
            file_db = load_file_database()
            if class_name in file_db:
                st.error("此代码已被使用，请选择一个新的代码。")
            else:
                store_docs(uploaded_file, class_name)
                st.success(f"文件已上传。文件代码：{class_name}")
    else:
        st.write("请先上传文件并输入代码")

    # 文件查询
    st.subheader("文件查询")
    file_db = load_file_database()
    class_names = list(file_db.keys())
    selected_file_code = st.selectbox("选择要查询的文件", class_names, key="query")

    if selected_file_code:
        file_info = file_db[selected_file_code]
        st.write(f"文件名: {file_info['name']}")
        
        # 添加下载按钮
        with open(file_info['path'], "rb") as file:
            st.download_button(
                label="下载文件",
                data=file,
                file_name=file_info['name'],
                mime="application/pdf"
            )

    # 文件删除
    st.subheader("文件删除")
    delete_file_code = st.selectbox("选择要删除的文件", class_names, key="delete")

    if delete_file_code:
        if st.button("删除文件"):
            if delete_file(delete_file_code):
                st.success(f"文件 {delete_file_code} 已成功删除")
                st.experimental_rerun()
            else:
                st.error("删除文件时出错")

if __name__ == '__main__':
    teacherPage()