#Hỏi đáp trên văn bản của mình có khi nó chưa đc train


from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf" #dữ liệu nó chưa đc train
vector_db_path = "vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Tao prompt template
def create_prompt(template):
    #Hãy dựa vào context sau trả lời câu hỏi
    prompt = PromptTemplate(template = template, input_variables=["context", "question"]) #context là các văn bản ta đã query vào trong vector db và nó đã lấy ra
    return prompt


# Tao simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff", #Hỏi đáp
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024), #Đưa ra 3 văn bản gần vs query
        return_source_documents = False, #Khong can dua ra cau trả lời thuộc văn bản nào ?
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

# Read tu VectorDB
def read_vectors_db():
    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db


# Bat dau thu nghiem
db = read_vectors_db()
llm = load_llm(model_file)

#Tao Prompt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = create_prompt(template)

llm_chain  =create_qa_chain(prompt, llm, db)

# Chay cai chain
question = "Ngày 18/12, SHB đã làm gì?"
response = llm_chain.invoke({"query": question})