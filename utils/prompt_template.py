class PromptTemplate:
    """
    A class for creating structured chat prompts by combining user queries with relevant document sources.

    Attributes:
        query (str): The user's input query
        response (str): The relevant document source text
        chat_prompt (str): The formatted chat prompt combining query and response
    """

    def __init__(self, query: str, response: str):
        """
        Initialize the PromptTemplate with a query and response.

        Args:
            query (str): The user's input query
            response (str): The relevant document source text
        """
        self.query = query
        self.response = response
        self.chat_prompt = self.create_prompt()

    def create_prompt(self) -> str:
        """
        Creates a formatted prompt combining the query and response.

        Returns:
            str: The formatted chat prompt
        """
        return f"""You are an assistant proficient in providing detailed and concise results based on the given user query and the relevant documents.
                Query: {self.query}
                Relevant Document Source: {self.response} \n\n
                """.strip()

    def get_prompt(self) -> str:
        """
        Returns the current chat prompt.

        Returns:
            str: The formatted chat prompt
        """
        return self.chat_prompt

    def update_query(self, new_query: str) -> None:
        """
        Updates the query and regenerates the chat prompt.

        Args:
            new_query (str): The new user query
        """
        self.query = new_query
        self.chat_prompt = self.create_prompt()

    def update_response(self, new_response: str) -> None:
        """
        Updates the response and regenerates the chat prompt.

        Args:
            new_response (str): The new document source
        """
        self.response = new_response
        self.chat_prompt = self.create_prompt()