from colpali_engine.models import ColQwen2, ColQwen2Processor
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from typing_extensions import TypedDict
from operator import add
from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END
from google import genai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import storage
from google.genai import types
import PIL
from io import BytesIO
import base64


# Configure Qdrant client
qdrantclient = QdrantClient(
    url="https://3bed5a5a-f523-4ae1-9ed0-d5b4408b8449.eu-west-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oC3xUAMljhPccXJQEZX1LjbXuO3CqYJ8CUwFZI2aGHk",
    timeout=100
)
collection_name = "dgmproject"

# Load the model
model_name = "vidore/colqwen2-v0.1"

device = "cpu"
print(f"Using device: {device}")

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Apple MPS doesn't support bfloat16
    device_map=device,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Set up Google API credentials
# IMPORTANT: You need to set these environment variables or load from a file
# Option 1: Set environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyA9I36rGn5QFV-GP8YyUvmBAzQaOsyELi4"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/saivignesh/Documents/DGM_Project/Browser-Extension/fabled-emblem-450414-r0-a631ebb8abd9.json"

# Initialize Google Cloud Storage client
try:
    storage_client = storage.Client()
    buckets = list(storage_client.list_buckets())
    print("Connected to Google Cloud Storage")
    print(f"Available buckets: {len(buckets)}")
except Exception as e:
    print(f"Failed to connect to Google Cloud Storage: {e}")
    print("Make sure your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly")

# Messages state class
class Messages(MessagesState):
    messages: Annotated[list[Any], add_messages]

# Tool function to query search database
def query_search_db(query: str):
    """
    Processes a user query to retrieve relevant documents from a vector database
    and generates a response using a Large Language Model (LLM).
    """
    if isinstance(query, str):
        query = [query]

    def batch_embed_query(query_batch, model_processor, model):
        with torch.no_grad():
            processed_queries = model_processor.process_queries(query_batch).to(model.device)

            print("Processed Images Type:", type(processed_queries))
            print(f"Processed Images Keys: {processed_queries.keys()}")

            query_embeddings_batch = model(**processed_queries)
        return query_embeddings_batch.cpu().float().numpy()

    colqwen_query = batch_embed_query(query, processor, model)

    LLM_category=['geeksforgeeks']

    # Search the Qdrant vector database
    def reranking_search_batch(query_batch,
                           collection_name,
                           search_limit=20,
                           prefetch_limit=200,
                           FilterList=LLM_category):
        filter_ = None

        # Apply filter only if FilterList is not empty
        if FilterList:
            filter_ = models.Filter(
                should=[
                    models.FieldCondition(
                        key="Book_Category",
                        match=models.MatchAny(any=FilterList)
                    )
                ]
            )

        search_queries = [
            models.QueryRequest(
                query=query,
                prefetch=[
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_columns"
                    ),
                    models.Prefetch(
                        query=query,
                        limit=prefetch_limit,
                        using="mean_pooling_rows"
                    ),
                ],
                filter=filter_,
                limit=search_limit,
                with_payload=True,
                with_vector=False,
                using="original"
            ) for query in query_batch
        ]

        response = qdrantclient.query_batch_points(
            collection_name=collection_name,
            requests=search_queries
        )

        # If no results, perform a broad search without filters
        if all(not res.points for res in response):
            search_queries = [
                models.QueryRequest(
                    query=query,
                    prefetch=[
                        models.Prefetch(
                            query=query,
                            limit=prefetch_limit,
                            using="mean_pooling_columns"
                        ),
                        models.Prefetch(
                            query=query,
                            limit=prefetch_limit,
                            using="mean_pooling_rows"
                        ),
                    ],
                    filter=None,  # Broad search
                    limit=search_limit,
                    with_payload=True,
                    with_vector=False,
                    using="original"
                ) for query in query_batch
            ]
            response = qdrantclient.query_batch_points(
                collection_name=collection_name,
                requests=search_queries
            )

        return response

    answer_colqwen = reranking_search_batch(colqwen_query, collection_name)

    top_10_results = []
    for point in answer_colqwen[0].points[:10]:
        top_10_results.append({"image_link": point.payload['image_link']})

    def download_gcs_image(gcs_uri):
        """Download an image from GCS and return its raw bytes."""
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI format: {gcs_uri}")

        bucket_name = gcs_uri.split("/")[2]
        blob_path = "/".join(gcs_uri.split("/")[3:])

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        return blob.download_as_bytes()  # Return raw image bytes

    # Convert GCS images to inline data format
    images_payload = [
        types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=download_gcs_image(result["image_link"])
            )
        )
        for result in top_10_results
    ]

    # Construct the prompt
    prompt_text = f"""
    You are a highly knowledgeable assistant with expertise in analyzing and synthesizing information.
    Below are relevant details (images, metadata) to answer the user's question accurately.

    User's Question:
    {query}

    Your task:
    - Analyze the images provided.
    - Use the metadata to generate an accurate and detailed response.
    - Avoid unrelated or speculative information.
    - Ensure the response is clear, concise, directly addresses the user's query.
    - Don't add your own points to the answer.
    """

    # Send request to Gemini Vision API
    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        print("Sending data to Gemini Vision API...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt_text)] + images_payload)]
        )
        return response.text
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return f"An error occurred while processing your request: {str(e)}"

# Set up the LLM
tools = [query_search_db]

def googleGenerativeAI(model="gemini-1.5-pro", temperature=0):
    """Create a ChatGoogleGenerativeAI instance"""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)

llm = googleGenerativeAI(model="gemini-1.5-pro", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content=
                  """ - You are a Helpful AI Agent.
                      - If you obtain a response from other llm then dont do anything, just check it relates to the user query and print the response to the user.
                      - Avoid unrelated or speculative information.
                      - Ensure the response is clear, concise, and directly addresses the user's query""")

# Assistant node
def assistant(state: MessagesState):
   return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}

# Build the graph
builder = StateGraph(MessagesState)

# Define nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Create memory and compile graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Function to visualize graph - optional for VSCode, requires graphviz
def visualize_graph():
    try:
        import graphviz
        from IPython.display import display
        
        # Save the graph to a file
        dot_data = graph.get_graph(xray=True).draw()
        
        # Create a Graphviz object
        g = graphviz.Source(dot_data)
        
        # Render to a file
        g.render("graph_visualization", format="png")
        print("Graph visualization saved as 'graph_visualization.png'")
        
        # For IPython in VSCode
        try:
            display(g)
        except:
            pass
    except ImportError:
        print("Graphviz not installed. Install with: pip install graphviz")
        print("Also install system-level Graphviz: https://graphviz.org/download/")

# Uncomment to visualize
visualize_graph()

# Example usage function
def run_query(user_query):
    config = {"configurable": {"thread_id": "1"}}
    messages = [HumanMessage(content=user_query)]
    result = graph.invoke({"messages": messages}, config)
    return result["messages"][-1].content

# Example usage
if __name__ == "__main__":
    response = run_query("Importing libraries required for DBSCAN Algorithm?")
    print(response)