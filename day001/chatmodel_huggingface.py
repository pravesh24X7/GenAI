from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Nanbeige/Nanbeige4.1-3B:featherless-ai",
                          task="text-generation",
                          max_new_tokens=500,)
llm_model = ChatHuggingFace(llm=llm)

model_response = response = llm_model.invoke(
    "Answer briefly: What is the capital of India?"
)
print(model_response.content)