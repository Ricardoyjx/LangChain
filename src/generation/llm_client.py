from langchain_ollama import ChatOllama


def create_ollama_client(model_name, base_url, temperature):
    return ChatOllama(
        name=model_name,
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )
