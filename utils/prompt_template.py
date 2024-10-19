class PromptTemplate:
    def __init__(self):
        pass
    chat_prompt = f"""
                    You are an assistant proficient in providing detailed and concise results based on the given user q
                    uery and the relevant documents: \n\n
                    query: {query} \n\n
                    relevant document source: {response}
    """


