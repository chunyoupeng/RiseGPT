import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from utils import *
import json
import os 
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

_template = """Answer the user questions based on the context. If you don't know, just 
say "我不知道".不要强行作答.
<context>
{context}
<context/>
"""
# Prompt

# conversation = ConversationChain(llm=chat_model)

# Start the conversation
os.environ["OPENAI_API_BASE"] = "https://aiapi.xing-yun.cn/v1" 
os.environ["OPENAI_API_KEY"] = "sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a"

st.title("RiseGPT")
with st.expander("ℹ️ 说明"):
    st.caption(
        "重庆市西南大学Rise实验室刘志明老师论文助手"
    )

# openai.api_key = st.secrets["OPENAI_API_KEY"]
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Maximum allowed messages
max_messages = (
    100  # Counting both user and assistant messages, so 10 iterations of conversation
)
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def remain_last(input_str):
    parts = input_str.split("/")
    desired_part = parts[-1]
    base_name, _ = os.path.splitext(desired_part)
    return base_name

def get_response(question, docs):

    context = "\n".join([doc.page_content for doc in docs])
    stream_handler = StreamHandler(st.empty()) 
    chat_model = ChatOpenAI(
            openai_api_base="https://aiapi.xing-yun.cn/v1",
            openai_api_key="sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a",
            streaming=True,  # ! important
            callbacks=[stream_handler], # ! important
            model_name="gpt-3.5-turbo"
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name

    list_history = st.session_state.messages
    list_history_str = json.dumps(list_history, ensure_ascii=False)
    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context({"input":"hi"},{"output": list_history_str})
    print(memory.load_memory_variables({}))
    # conversation = LLMChain(llm=chat_model, prompt=prompt, verbose=True, memory=memory)
    chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | chat_model
    )

    response = chain.invoke({"input": question, "context": context})
    return response.content

persist_directory = "data/vector_src/lzm-vectorstore"
if len(st.session_state.messages) >= max_messages:
    st.info(
        "您的使用次数过多了，请休息一下，休息完后，请重新打开网页，继续使用"
    )

else:
    if question := st.chat_input("请输入您对刘志明老师文章的疑问，希望能帮您解惑"):
        docs = get_documents(question)

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            full_response = ""
            response = get_response(question, docs)
            # print(response)
            full_response += response
            # message_placeholder.markdown(full_response + "▌")
            # message_placeholder.markdown(full_response)
            sources = "\n\n".join([f"📚 来源: { remain_last( d.metadata['source'] ) } 第 {d.metadata['page']}页" for d in docs])
            message_placeholder = st.empty()
            message_placeholder.markdown(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        # print(st.session_state.messages)
