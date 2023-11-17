from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain.document_loaders import PyPDFLoader
import os
import tiktoken  # !pip install tiktoken
import sys
import re
import os





def load_pdf(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"Begin loading {filename}")
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            document = loader.load()
            documents.extend(document)
            print(f"{filename} load successfully")
    return documents


def delete_space(path):
    import os

    # 指定文件夹路径
    folder_path = path

    # 遍历文件夹中的所有文件
# 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 只处理.txt文件
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # 打开文件并读取内容
                with open(file_path, 'r') as f:
                    content = f.read()

                # 去掉所有的空格和换行符
                processed_content = content.replace(' ', '').replace('\n', '')

                # 重新写入文件
                with open(file_path, 'w') as f:
                    f.write(processed_content)

    print("处理完成!")



def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('p50k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def clean_string(input_doc):
    placeholder = "<DOUBLE_NEWLINE>"
    step1 = input_doc.page_content.replace("\n\n", placeholder)
    step2 = step1.replace("\n", "").replace(" ", "").replace(".", "").replace('\x00', "")
    result = step2.replace(placeholder, "\n\n")
    cleaned_text = re.sub('[\ue000-\uf8ff]', '', result)
    final_text = re.sub(r'\s+', "", cleaned_text)
    threshold = 10
    cleaned_content = re.sub(
        r'[A-Za-z0-9$%!@#^&*]{' + str(threshold) + ',}', '', final_text)
    input_doc.page_content = cleaned_content
    input_doc.metadata = input_doc.metadata
    return input_doc



def ingest():

    PATH = "data/vector_src"
    print("Loading data...")
    # loader = UnstructuredFileLoader("state_of_the_union.txt")

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    persist_directory = PATH + "/" +  "lzm-vectorstore/"
    folder_path = "data/docs/" + "lzm/"  # path to xx_src
    loader = PyPDFDirectoryLoader(folder_path)
    text_docs = DirectoryLoader(path=folder_path, glob="**/*.txt").load()
    # delete_space(folder_path)
    # raw_documents = load_pdf(sys.argv[1])
    raw_documents = loader.load()
    raw_documents.extend(text_docs)

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=50,
        length_function=tiktoken_len,
    )
    documents = text_splitter.split_documents(raw_documents)

    documents = list(map(lambda doc: clean_string(doc), documents))
    print(documents[0])
    print("Creating vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    # vectorstore = FAISS.from_documents(documents, embeddings)

    # with open(DIR_SAVED, "wb") as f:
    #     pickle.dump(vectorstore, f)


def get_documents(question):
    persist_directory = "data/vector_src/lzm-vectorstore"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriver = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 3})
    docs = retriver.get_relevant_documents(query=question)
    return docs

            