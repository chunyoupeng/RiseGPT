from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# Create a chat model
chat_model = ChatOpenAI(openai_api_base="https://aiapi.xing-yun.cn/v1",
        openai_api_key="sk-3e5wTBAl2iFDvQvW9b5693C90a97425eBf3b4bEa558eC66a",
        model_name="gpt-3.5-turbo")

# Create a conversation chain with the chat model
conversation = ConversationChain(llm=chat_model)

# Start the conversation
response = conversation.run("Hi, how can I assist you?")

# Print the response
print(response)
