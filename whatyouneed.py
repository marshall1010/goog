CHUNK_SIZE = 1096
CHUNK_OVERLAP = 10
FETCH_COUNT = 2
NUM_PROBLEMS = 5
SESSION_ID = "Default"
DIRECTORY = "./"
LANGUAGE = "english"

import os
import sys
import streamlit as st
import requests
import sqlite3
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
import pandas as pd
from io import BytesIO
import json
import uuid

llm = ChatOpenAI(temperature=0.7, model_name='gpt-4-0125-preview')
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

store = {}

def process_file(uploaded_file):
    documents = []

    if uploaded_file is not None:
        # 保存上傳的文件到臨時目錄
        temp_dir = "/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 載入上傳的文件
        loader = PyPDFLoader(temp_file_path)
        documents.extend(loader.load())

        # 刪除臨時文件
        os.remove(temp_file_path)

    # 分割文檔內容
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = text_splitter.split_documents(documents)

    # 創建向量存儲
    vector_store = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")

    # 創建檢索器
    retriever = vector_store.as_retriever(search_kwargs={'k': FETCH_COUNT})

    return retriever

def custom_create_retrieval_chain(llm, retriever, contextualize_q_prompt):
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    q_and_a_system_prompt = '''You are a helpful Teacher's Assistant answering a student's questions.
            Use the following pieces of retrieved context to answer the question in {language} to
            the best of your ability.
            If you cannot answer a question respond with "對不起，這個我不知道。請問你的老師 /
            {context}"'''
    q_and_a_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", q_and_a_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    q_and_a_chain = create_stuff_documents_chain(llm, q_and_a_prompt)

    retrieval_chain = create_retrieval_chain(history_aware_retriever, q_and_a_chain)
    return retrieval_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def set_session_id(userID):
    global SESSION_ID
    SESSION_ID = userID

# 產生問題的函數
def give_problems():
    problem_prompt = f"Please generate {NUM_PROBLEMS} problem-solving questions testing the information contained in the documents below. For each question, list the document that it relates to."
    return invoke(problem_prompt)

def invoke(text_input):
    response = st.session_state.conversational_rag_chain.invoke(
        {"input": text_input,
         "language": LANGUAGE,
        },
        config={"configurable": {"session_id": SESSION_ID}}
    )
    answer = "\n".join([response["answer"], "\n The source document names are listed here:"]
                       + [doc.metadata.get("source") for doc in response["context"]])
    return answer

# 設定 session_state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'role' not in st.session_state:
    st.session_state.role = ''
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'score' not in st.session_state:
    st.session_state.score = '0'

# 登入頁面
def login_page():
    st.title('Login Interface with Streamlit and Flask')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    role = st.selectbox('Role', ['teacher', 'student'])

    if st.button('Login'):
        response = requests.post('http://localhost:5000/login', json={'username': username, 'password': password, 'role': role})
        data = response.json()
        if data['status'] == 'success':
            st.session_state.logged_in = True
            st.session_state.role = data['role']
            st.experimental_rerun()
        else:
            st.error(data['message'])

# 老師的頁面
def teacher_page():
    UPLOAD_DIR = "uploaded_files"
    DB_FILE = "file_database.json"
    if not os.path.exists(UPLOAD_DIR):
       os.makedirs(UPLOAD_DIR)
 
# 加载或创建文件数据库
    def load_file_database():
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        return {}

# 保存文件数据库
    def save_file_database(db):
        with open(DB_FILE, 'w') as f:
             json.dump(db, f)

# 生成唯一的文件代号
    def save_uploaded_file(uploaded_file, file_code):
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
             f.write(uploaded_file.getbuffer())
    
        file_db = load_file_database()
        file_db[file_code] = {
             "name": uploaded_file.name,
             "path": file_path
        }
        save_file_database(file_db)

# 删除文件和数据库条目
    def delete_file(file_code):
        file_db = load_file_database()
        if file_code in file_db:
            file_path = file_db[file_code]['path']
            if os.path.exists(file_path):
               os.remove(file_path)
            del file_db[file_code]
            save_file_database(file_db)
            return True
        return False

# Streamlit 应用
    st.title("文件上传、查询和删除系统")

# 文件上传部分
    uploaded_file = st.file_uploader("选择要上传的文件", type=['pdf', 'txt', 'doc', 'docx'])
    file_code = st.text_input("为文件指定一个唯一代码")
    if uploaded_file is not None and file_code:
           if st.button("上传文件"):
              file_db = load_file_database()
              if file_code in file_db:
                 st.error("此代码已被使用，请选择一个新的代码。")
              else:
                save_uploaded_file(uploaded_file, file_code)
                st.success(f"文件已上传。文件代码：{file_code}")

# 文件检索部分
    st.subheader("文件查询")
    file_db = load_file_database()
    file_codes = list(file_db.keys())
    selected_file_code = st.selectbox("选择要查询的文件", file_codes, key="query")
    if selected_file_code:
       file_info = file_db[selected_file_code]
       st.write(f"文件名: {file_info['name']}")
    
    # 添加下载按钮
    with open(file_info['path'], "rb") as file:
        st.download_button(
            label="下载文件",
            data=file,
            file_name=file_info['name'],
            mime="application/octet-stream"
        )
    st.subheader("文件删除")
    delete_file_code = st.selectbox("选择要删除的文件", file_codes, key="delete")
    if delete_file_code:
           if st.button("删除文件"):
               if delete_file(delete_file_code):
                  st.success(f"文件 {delete_file_code} 已成功删除")
                  st.experimental_rerun()
    else:
        st.error("删除文件时出错")

# 显示所有文件
        st.subheader("所有上传的文件")
        file_db = load_file_database()
        for code, info in file_db.items():
            st.write(f"代码: {code}, 文件名: {info['name']}")
# 學生的頁面
def student_page():
    st.title('Student Page')
    st.write('Welcome, Student!')
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.role = ''
        st.experimental_rerun()

# 根據登入狀態顯示不同頁面
if st.session_state.logged_in:
    if st.session_state.role == 'teacher':
        teacher_page()
    elif st.session_state.role == 'student':
        student_page()
else:
    login_page()
    teacher_page()
