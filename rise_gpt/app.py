import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

chat_model = ChatOpenAI(
        openai_api_base="https://aiapi.xing-yun.cn/v1",
        openai_api_key="sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a",
        streaming=True,  # ! important
        callbacks=[StreamingStdOutCallbackHandler()], # ! important
        model_name="gpt-3.5-turbo"
    )
_template = """Answer the user questions based on the context
<context>
<context/>
"""
# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(llm=chat_model, prompt=prompt, verbose=True, memory=memory)

# conversation = ConversationChain(llm=chat_model)

# Start the conversation

st.title("RiseGPT")
with st.expander("ℹ️ 说明"):
    st.caption(
        "重庆市西南大学Rise实验室"
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
    1000  # Counting both user and assistant messages, so 10 iterations of conversation
)

if len(st.session_state.messages) >= max_messages:
    st.info(
        """Notice: The maximum message limit for this demo version has been reached. We value your interest!
        We encourage you to experience further interactions by building your own application with instructions
        from Streamlit's [Build conversational apps](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)
        tutorial. Thank you for your understanding."""
    )

else:
    if prompt := st.chat_input("请输入您对刘志明老师文章的疑问，希望能帮您解惑"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = conversation({"question": prompt})
            print(response)
            full_response += response["text"]
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
