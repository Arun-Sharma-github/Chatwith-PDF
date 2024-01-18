from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from langchain.chains import LLMChain
#**Step 1: Load the PDF File from Data Path****
loader=DirectoryLoader('data/',
                       glob="data.pdf",
                       loader_cls=PyPDFLoader)

documents=loader.load()


#print(documents)

#***Step 2: Split Text into Chunks***

text_splitter=RecursiveCharacterTextSplitter(
                                             chunk_size=200,
                                             chunk_overlap=20)


text_chunks=text_splitter.split_documents(documents)

# print(len(text_chunks))
#**Step 3: Load the Embedding Model***


embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})


#**Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
vector_store=FAISS.from_documents(text_chunks, embeddings)


##**Step 5: Find the Top 3 Answers for the Query***

# query="Business and Personality Develeopment"

# docs = vector_store.similarity_search(query)

# print(docs)
llm=CTransformers(model="models\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':128,
                          'temperature':0.1})

template="""
Answer short and precise
I am a helpful, respectful and honest Business and Personality Develeopment Coach. 
Do not answer anything beyond Business and Personality Develeopment Coaching just say I have expertise in Business and Personality Develeopment and can't answer this question.

Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 

Use the following pieces of information to answer the user's question.

Do not refer to external links or resources in your answer.
Summarize your answer in 50-100 words.

avoid repeating the same set of words in your answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

#start=timeit.default_timer()

chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})

# LLM_Chain=LLMChain(prompt=qa_prompt, llm=llm)


def api(ques):
    print(ques)
    response=chain({"query":ques})
    print(response)
    return response['result']
    


#end=timeit.default_timer()
#print(f"Here is the complete Response: {response}")

#print(f"Here is the final answer: {response['result']}")

#print(f"Time to generate response: {end-start}")

# while True:
#     user_input=input(f"prompt:")
#     if user_input=='exit':
#         print('Exiting')
#         sys.exit()
#     if user_input=='':
#         continue
#     result=chain({'query':user_input})
#     print(f"Answer:{result['result']}")