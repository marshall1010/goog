import streamlit as st
import requests
import os
import sys
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
from IPython.display import display, Markdown
from openai import OpenAI
import re
os.environ["OPENAI_API_KEY"] =

CHUNK_SIZE = 1096
CHUNK_OVERLAP = 1096
FETCH_COUNT = 2
NUM_PROBLEMS = 1
SESSION_ID = "Default"
DIRECTORY = "./"
LANGUAGE = "繁體中文"

chunk_size = 1096
chunk_overlap = 10
fetch_count = 2
num_problems = 1
session_id = "Default"
directory = "./"
language = "繁體中文"



# 保存上传的文件
def save_uploaded_file(uploaded_file):
    with open(os.path.join(DIRECTORY, uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())

# 处理文档函数
def process_documents(directory):
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(directory, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = os.path.join(directory, file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = os.path.join(directory, file)
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    # 文档分割
    chunk_size = 1096 # 可以根据需要调整
    chunk_overlap = 10  # 可以根据需要调整
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # 创建向量存储
    vector_store = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")

    # 获取检索器
    fetch_count =  2  # 设置检索的数量
    retriever = vector_store.as_retriever(search_kwargs={'k': fetch_count})

    return retriever


retriever = process_documents(directory)
# Contextualize question with chat history and create retrieval chain 對不起，我這個不知道怎麼翻 ：（
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

# Create a retriever to find related documents 對不起，我這個不知道怎麼翻 ：（
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

# 學生問題的樣子
q_and_a_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", q_and_a_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

q_and_a_chain = create_stuff_documents_chain(llm, q_and_a_prompt)

retrieval_chain = create_retrieval_chain(history_aware_retriever, q_and_a_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def set_session_id(userID):
  global SESSION_ID
  SESSION_ID = userID
def invoke(text_input):

  response = conversational_rag_chain.invoke(
      {"input": text_input,
         "language" : LANGUAGE,
         },
        config = {"configurable": {"session_id": SESSION_ID}}
        )
  answer = "\n".join([response["answer"], "\n The source document names are listed here:"]
                     + [doc.metadata.get("source") for doc in response["context"]])
  return answer
# 出問題的 function
def give_problems():
  problem_prompt = f"Please generate {NUM_PROBLEMS} problem-solving questions testing the information contained in the documents below. For each question, list the document that it relates to."
  return invoke(problem_prompt)

# Lanchain to combine retrieval chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

client = OpenAI(
                api_key =
)
def openai_api(prompt):
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": prompt}]
      )
  return completion.choices[0].message.content

def extract_score(text):
    # 定義正則表達式模式來匹配不同的樣式，包括方括號中的數字
    pattern = r'(\d+分|\[(\d+)\]|\b評分(\d+)分\b)'

    # 查找所有匹配的樣式
    matches = re.findall(pattern, text)

    # 提取數值
    scores = []
    for match in matches:
        for group in match:
            if group.isdigit():
                scores.append(int(group))
            elif '分' in group:
                scores.append(int(re.findall(r'\d+', group)[0]))

    # 去除括號並提取數值
    scores = [int(re.sub(r'[^\d]', '', str(score))) for score in scores]

    return scores



def calculate_weighted_sum(scores, weightings):
    total = 0
    for i in range(len(scores)):
        # 提取分数，如果是列表则取第一个元素，否则直接取值
        score_value = scores[i][0] if isinstance(scores[i], list) else scores[i]
        # 加权计算
        total += score_value * weightings[i]
    return total


def evaluate_student_answer(student_answers, question):

      score = [0 for _ in range(10)]
      #weighting =  [40,0,0,0,0,0,0,0,0,0]
      weighting =  [40,20,10,10,0,0,0,0,20,0]
      qbody = question
      appendix_cstandard_3 = "教學內容"
      print(f"學生的答案内容：{student_answers}")
      sanswer = student_answers
      cstandard = ["1.內容的相關性：\
      (評分描述)回答與問題關聯極小或沒有關聯：1分 \
      (評分描述)回答與問題有少許關聯，但大部分內容不相關：2分 \
      (評分描述)回答在一定程度上回應了問題，但部分重要內容缺失或不正確：3分 \
      (評分描述)回答與問題高度相關，並且涵蓋了大部分重要內容：4分 \
      (評分描述)回答與問題高度相關，全面且精準地涵蓋了所有重要內容：5分 \
      ",
      "2.邏輯清晰性：(前提：考量字數勿過少(如兩三句話)，再評判內容邏輯清晰性者，才能獲得高分)\
      (評分描述)思路雜亂無章，缺乏邏輯性：1分 \
      (評分描述)部分思路可辨，但邏輯關聯薄弱：2分 \
      (評分描述)思路基本清晰，但邏輯鏈條有斷裂：3分 \
      (評分描述)思路清晰，邏輯連貫，但仍有少許跳躍：4分 \
      (評分描述)思路非常清晰，邏輯連貫，易於理解：5分 \
      ",
      "3.深度和複雜性：\(前提：答題要求字數不多、評分請考量字數的影響，合理調整您的期望值)\
      (評分描述)表面化，沒有深入分析或解釋：1分 \
      (評分描述)有嘗試深入分析或解釋，但非常有限：2分 \
      (評分描述)表現出一定的深度和複雜性，但未完全探討：3分 \
      (評分描述)深度和複雜性良好，提供了深入的分析和合理的解釋：4分 \
      (評分描述)非常深入且複雜，展示出高水平的分析和批判性思考：5分 \
      ",
      "4.事實準確性：\
      (評分描述)多數事實錯誤或假設無效：1分 \
      (評分描述)一些事實正確，但仍有多處錯誤：2分 \
      (評分描述)大部分事實正確，有少數錯誤：3分 \
      (評分描述)事實基本準確，偶有瑕疵：4分 \
      (評分描述)所有事實都是準確的，顯示出優秀的知識掌握：5分 \
      ",
      "5.創新性和原創性：\
      (評分描述)缺乏創新和原創思考，完全是重述他人觀點：1分 \
      (評分描述)稍有創新或原創思考，但非常有限：2分 \
      (評分描述)展示出一定的創新和原創思考：3分 \
      (評分描述)論述中有明顯的創新和原創元素：4分 \
      (評分描述)論述極富創新性和原創性，提出了新觀點或獨到見解：5分 \
      ",
      "6.結構和組織：\
      (評分描述)結構雜亂無章，缺乏明確的組織：1分 \
      (評分描述)結構和組織可見但較弱，訊息有時顯得零散：2分 \
      (評分描述)結構合理，訊息組織基本清晰：3分 \
      (評分描述)結構清晰，訊息組織有條理，順暢連貫：4分 \
      (評分描述)結構優秀，訊息組織得宜，清晰，流暢：5分 \
      ",
      "7.證據的使用：\
      (評分描述)未使用證據或例子支持觀點：1分 \
      (評分描述)僅提供了有限或不相關的證據支持觀點：2分 \
      (評分描述)提供了一些相關證據，但不是很充分：3分 \
      (評分描述)適當地使用了充分、相關的證據支持觀點：4分 \
      (評分描述)使用了豐富、相關且有力的證據優秀地支持觀點：5分 \
      ",
      "8.語言和風格：\
      (評分描述)語言混亂，大量語法和拼寫錯誤：1分 \
      (評分描述)語言表達有所困難，語法和拼寫錯誤較多：2分 \
      (評分描述)語言表達尚清晰，有少量語法或拼寫錯誤：3分 \
      (評分描述)語言表達清晰，風格適當，幾乎沒有語法或拼寫錯誤：4分 \
      (評分描述)語言流暢，風格一致，無語法或拼寫錯誤：5分 \
      ",
      "9.批判性思維：\
      (評分描述)沒有表現出批判性思維，缺乏對問題的深入理解：1分 \
      (評分描述)有限的批判性思維，僅在表面上分析問題：2分 \
      (評分描述)表現出一定程度的批判性思維，提出了問題的某些假設：3分 \
      (評分描述)良好的批判性思維，深入分析問題並質疑假設：4分 \
      (評分描述)優秀的批判性思維，深刻分析和質疑問題及其假設：5分 \
      ",
      "10.綜合和整合能力：\
      (評分描述)沒有將訊息或觀點進行有效綜合或整合：1分 \
      (評分描述)對訊息或觀點進行了有限的綜合或整合：2分 \
      (評分描述)能夠將訊息或觀點進行基本的綜合和整合：3分 \
      (評分描述)很好地綜合和整合了訊息或觀點，展示了聯繫：4分 \
      (評分描述)卓越地綜合和整合了訊息或觀點，表現出高度的洞察力：5分 \
      "
      ,
      "Zh-TW"
      ]
      if weighting[0] != 0:
        print("輸出問題")
        print(qbody)
        print("學生答案")
        print(sanswer)
        print("評分標準")
        print(cstandard)
        prompt1 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答。討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[0] + "。\
        請依據批改評分：1-5分，只需要有分數即可，不用其他評論。回覆格式：分數"
        score_1 = openai_api(prompt1)
        score[0] = extract_score(score_1)
        while len(str(score[0]))!=3:
              score_1 = openai_api(prompt1)
              score[0] = extract_score(score_1)
      else:
        print("第一階段不評分")

      #第二階段：邏輯清晰性(考量字數)
      if weighting[1] != 0:
        prompt2 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。學生作答字數：" + str(len(sanswer)) +"。批改原則參考： " + cstandard[1] + "。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_2 = openai_api(prompt2)
        score[1] = extract_score(score_2)
        while len(str(score[1]))!=3:
              score_2 = openai_api(prompt2)
              score[1] = extract_score(score_2)
      else:
        print("第二階段不評分")

      #第三階段：深度和複雜性(考量字數)
      if weighting[2] != 0:
        prompt3 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。學生作答字數：" + str(len(sanswer)) +"批改原則參考： " + cstandard[2] + "。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_3 = openai_api(prompt3)
        score[2] = extract_score(score_3)
        while len(str(score[2]))!=3:
              score_3 = openai_api(prompt3)
              score[2] = extract_score(score_3)
      else:
        print("第三階段不評分")

      #第四階段：事實準確性，提供「事實描述」做比對(只有這個不同)
      if weighting[3] != 0:
        prompt4 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。課堂內容：" + appendix_cstandard_3 + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[3] + "。\
        事實描述： " + appendix_cstandard_3 + "\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_4 = openai_api(prompt4)
        score[3] = extract_score(score_4)
        while len(str(score[3]))!=3:
              score_4 = openai_api(prompt4)
              score[3] = extract_score(score_4)
      else:
        print("第四階段不評分")

      #第五階段：創新性和原創性
      if weighting[4] != 0:
        prompt5 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[4] + "。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_5 = openai_api(prompt5)
        score[4] = extract_score(score_5)
        while len(str(score[4]))!=3:
              score_5 = openai_api(prompt5)
              score[4] = extract_score(score_5)
      else:
        print("第五階段不評分")

      #第六階段：結構和組織
      if weighting[5] != 0:
        prompt6 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[5] + "。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_6 = openai_api(prompt6)
        score[5] = extract_score(score_6)
        while len(str(score[5]))!=3:
              score_6 = openai_api(prompt6)
              score[5] = extract_score(score_6)
      else:
        print("第六階段不評分")

      #第七階段：證據的使用
      if weighting[6] != 0:
        prompt7 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[6] + "。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_7 = openai_api(prompt7)
        score[6] = extract_score(score_7)
        while len(str(score[6]))!=3:
              score_7 = openai_api(prompt7)
              score[6] = extract_score(score_7)
      else:
        print("第七階段不評分")

      #第八階段：語言和風格
      if weighting[7] != 0:
        prompt8 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[7] + "。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_8 = openai_api(prompt8)
        score[7] = extract_score(score_8)
        while len(str(score[7]))!=3:
              score_8 = openai_api(prompt8)
              score[7] = extract_score(score_8)
      else:
        print("第八階段不評分")

      #第九階段：批判性思維
      if weighting[8] != 0:
        prompt9 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[8] + "。請依據批改原則裡的評分描述回覆：1-5分。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_9 = openai_api(prompt9)
        score[8] = extract_score(score_9)
        while len(str(score[8]))!=3:
              score_9 = openai_api(prompt9)
              score[8] = extract_score(score_9)
      else:
        print("第九階段不評分")

      #第十階段：綜合和整合能力
      if weighting[9] != 0:
        prompt10 = "請扮演討論區批改教師，下列提供討論區題目及學生的作答，請注意下列評分原則。討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cstandard[9] + "。\
        請依據批改評分：1-5分。回覆評分：分數。只需要有分數即可，不用其他評論"
        score_10 = openai_api(prompt10)
        score[9] = extract_score(score_10)
        while len(str(score[9]))!=3:
              score_10 = openai_api(prompt10)
              score[9] = extract_score(score_10)
      else:
        print("第十階段不評分")

      total_score = calculate_weighted_sum(score, weighting)
      print("總分：", total_score/5)


      #total_ans_length = len(sanswer)
      #print("字數：", total_ans_length)

      # 使用sorted()函数结合enumerate()函数获取索引并按值排序
      # reverse=True确保列表是按降序排列的
      sorted_indices = sorted(range(len(weighting)), key=lambda i: weighting[i], reverse=True)
      # 获取前三大数值的索引
      top_three_indices = sorted_indices[:5]

      select_item = [1,2,3,4,9] #共10個指標,選5個
      cst = [cstandard[select_item[0]-1],cstandard[select_item[1]-1],cstandard[select_item[2]-1],cstandard[select_item[3]-1],cstandard[select_item[4]-1]]\
      # 將這5個指標連接成一個字符串
      cst_joined = "；".join(cst)
      print(cst_joined)

      #第十一階段：整合評語
      # 檢查指標4是否在選擇的指標中
      appendix = ""
      if 4 in select_item:
          appendix = "課堂內容：" + appendix_cstandard_3 + "\n"

      prompt11 = "請扮演數位課程討論區批改教師，署名「工研院AI智慧助教」，我會給您每個學生的作答評論參考，請您按照以下步骤完成評語，回覆給每一個學生：\
      討論區題目：" + qbody + "。學生作答：" + sanswer +"。批改原則參考： " + cst_joined +  "。\
      若評分指標有「事實準確性」請依照「事實描述」做比對，事實描述： " + appendix_cstandard_3 + "。\
      原則1：編寫一段有結構的評語文章（約300字，包含引言、正文、結語）。\
      原則2：在評語中引用學員的一句話，作為討論的起點。\
      原則3：評語針對「主要評分項目」評論，提供大量具體的鼓勵和肯定，少量聚焦於一個主要改進領域。\
      原則4：勿評論分數、字數、標點符號。請避免使用強烈的負面轉折詞，如「然而」、「但是」。\
      原則5：回覆用Zh-TW。\
      "
      comment = openai_api(prompt11);
      display(Markdown(comment));

      return {"您的分數為": total_score/5},comment

# 初始化 session_state 中的变量
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'role' not in st.session_state:
    st.session_state.role = ''

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
    st.title('Teacher Page')
    st.write('Welcome, Teacher!')
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        st.write("filename:", uploaded_file.name)
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.role = ''
        st.experimental_rerun()
# 學生的頁面
def show_stundent_page():
    st.title('Student Page')
    st.write('Welcome, Student!')
    if st.button('Generate Problem'):
         question = give_problems()  # 可以传递一个现有问题的列表
         st.session_state["question"] = question
         st.text_area("Question", question, height=100)
         question = st.text_input("question")
         answer1 = st.text_input("answer")
    if st.button('Score'):
         [score, score_text] = evaluate_student_answer(answer1, question)
         st.text_area("Score", score, height=100)
         st.text_area("Feedback", score_text, height=100)
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.role = ''
        st.experimental_rerun()
 
if st.session_state.logged_in:
    if st.session_state.role == 'teacher':
        teacher_page()
    elif st.session_state.role == 'student':
        show_stundent_page()
else:
    login_page()
