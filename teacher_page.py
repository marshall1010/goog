import streamlit as st
from mymath import openai_api
from functions_file import *
from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from io import StringIO
import json
from PyPDF2 import PdfReader
import firebase_admin
from firebase_admin import credentials, firestore


PROJECT_ID = "for-test-4cb36"
os.environ["OPENAI_API_KEY"] = ""
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "for_test_credentials.json"
#set GOOGLE_APPLICATION_CREDENTIALS= "for_test_credentials.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
DB_FILE = "for_test_credentials.json"
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

def store_docs(file, class_name):
    documents = []

    if UPLOAD_DIR is not None:
        # 保存上傳的文件到臨時目錄
        temp_dir = "/tmp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_file_path = os.path.join(temp_dir, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())

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

def teacher_page():
    st.title('Teacher Page')
    st.write('Welcome, Teacher!')

    # 文件上传
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

    # 显示所有文件
    st.subheader("所有上传的文件")
    for code, info in file_db.items():
        st.write(f"代码: {code}, 文件名: {info['name']}")

if __name__ == '__main__':
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    teacher_page()