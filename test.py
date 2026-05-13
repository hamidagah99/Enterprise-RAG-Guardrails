from langchain_openai import ChatOpenAI

# This connects directly to your LM Studio server
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio", # It just needs a placeholder text here
    temperature=0.7
)

response = llm.invoke("Say exactly 'Connection successful!' if you can hear me.")
print(response.content)