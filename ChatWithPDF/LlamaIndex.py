import os
import torch
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser


file_path = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(file_path, '..', 'db')
DOCUMENT_DIR = os.path.join(file_path, '..', 'documents')
NEW_DOCUMENTS_DIR = os.path.join(file_path, '..', "new_documents")


class LlamaIndex:
    """
    A class to manage document indexing and retrieval using the LlamaIndex framework with Chroma as the vector store.

    Attributes:
        collection_name (str): The name of the Chroma collection for storing vectors.
        device (str): The device (cpu or cuda) for running the embedding model.
        _embed_model (HuggingFaceEmbedding): Class variable-The embedding model used for generating vector embeddings.
    """
    _embed_model = None
    def __init__(self, collection_name_) -> None:
        """
        Initializes the LlamaIndex class with the specified collection name and sets up the embedding model.

        Args:
            collection_name_ (str): The name of the collection to be used for storing vectors.
        """
        self.collection_name = collection_name_
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if LlamaIndex._embed_model is None:
            LlamaIndex._embed_model = HuggingFaceEmbedding(
                model_name="nomic-ai/nomic-embed-text-v1",
                trust_remote_code=True,
                device=self.device,
            )

    @staticmethod
    def document_parser(dir_path : str) -> list[BaseNode]:
        """
        Receives the document directory to be parsed and returns the parsed nodes

        Steps;
            1. Get the directory containing documents to be parsed
            2. Load the document via SimpleDirectoryLoader
            3. Using SimpleNodeParser parse the documents based on the chunk_size provided

        Args:
            dir_path (str): The path of the directory containing documents

        :return:
            base_node (list[BaseNode]): A list of parsed documents as nodes
        """
        documents_ = SimpleDirectoryReader(dir_path).load_data(show_progress=True, num_workers=-1)
        print(f"Loaded {len(documents_)} documents_")

        node_parser = SimpleNodeParser.from_defaults(chunk_size=1000, chunk_overlap=100)
        base_node = node_parser.get_nodes_from_documents(documents=documents_)
        print(f"Number of nodes parsed: {len(base_node)}")

        return base_node

    def index_documents(self) -> None:
        """
        Indexes documents by loading data from a directory, converting documents into nodes, and building the vector store.

        Steps:
            1. Load documents from the directory specified by DOCUMENT_DIR.
            2. Convert documents into nodes based on chunk_size using SimpleNodeParser.
            3. Initialize a ChromaDB client with the collection name.
            4. Build the VectorStore with the parsed nodes and save the embeddings.

        Returns:
            None
        """
        base_node = self.document_parser(dir_path=DOCUMENT_DIR)

        # 3
        db_client = chromadb.PersistentClient(path=DB_DIR)
        chroma_collection = db_client.get_or_create_collection(name=self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create a storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 4
        VectorStoreIndex(nodes=base_node,
                         storage_context=storage_context,
                         embed_model=LlamaIndex._embed_model)

    def retrieval_index(self) -> VectorStoreIndex:
        """
        Retrieves an initialized instance of VectorStoreIndex for querying purposes.

        Steps:
            1. Initialize a ChromaDB client and load the collection created earlier.
            2. Return the initialized VectorStoreIndex instance for retrieval purposes.

        Returns:
            VectorStoreIndex: An instance of VectorStoreIndex configured with the stored vectors and the embedding model.
        """
        # 1
        db_client = chromadb.PersistentClient(path=DB_DIR)
        chroma_collection = db_client.get_collection(name=self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        retrieval_indexer = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=LlamaIndex._embed_model
        )
        # 2
        return retrieval_indexer

    def insert_documents(self) -> None:
        """
        Inserts newly provided documents into the existing VectorStore

        Steps:
            1. Get the documents to be indexed
            2. Chunk the documents and make them into nodes
            3. Get the VectorStore initialized and then insert into the VectorStore using the similar configurations

        Args:

        :return: None
        """
        # 1.
        new_base_nodes = self.document_parser(dir_path=NEW_DOCUMENTS_DIR)
        # 2.
        insert_index = self.retrieval_index()
        insert_index.insert_nodes(nodes=new_base_nodes,
                                         show_progress=True)

        print(f"New Documents total: {len(new_base_nodes)} has been inserted")
        print("-" * 20)
        print(f"Total number of nodes in VectorStore: {insert_index.summary} ")

    def query_response(self, _query: str) -> list[NodeWithScore]:
        """
        Queries the indexed documents and retrieves the most similar response based on the query provided.

        Steps:
            1. Get the initialized instance from 'retrieval_index()'.
            2. Create the retrieval engine from the instance.
            3. Retrieve responses based on the similarity score using the query parameter.

        Args:
            _query (str): The query string provided by the user.

        Returns:
            List[Response]: A list of response objects containing the most similar documents based on the query.
        """
        # 1
        retrieval_index = self.retrieval_index()
        # 2
        retrieval_engine = retrieval_index.as_retriever(similarity_top_k=1)
        # 3
        query_responses = retrieval_engine.retrieve(_query)
        return query_responses

