import torch
import uuid
import os
import time
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path
from google.cloud import storage
import requests
from urllib.parse import urlparse
from weasyprint import HTML

# Import ColPali engine dependencies
from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import LangChain dependencies
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import Annotated, Any
from langgraph.checkpoint.memory import MemorySaver
from google import genai
from google.genai import types

class Messages(MessagesState):
    messages: Annotated[list[Any], add_messages]

# Configuration variables
QDRANT_URL = "https://3bed5a5a-f523-4ae1-9ed0-d5b4408b8449.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oC3xUAMljhPccXJQEZX1LjbXuO3CqYJ8CUwFZI2aGHk"
COLLECTION_NAME = "dgmproject"
GCS_BUCKET_NAME = "dgmtest"
MODEL_NAME = "vidore/colqwen2-v0.1"

def initialize_model():
    device = "cpu"
    print(f"Using device: {device}")

    model = ColQwen2.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=device,
    ).eval()
    
    processor = ColQwen2Processor.from_pretrained(MODEL_NAME)
    
    return model, processor

def initialize_qdrant_client():
    """Initialize the Qdrant client"""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=100
    )

def url_to_gcs_folder(content_url):
    """Convert URL to a valid GCS folder name"""
    return content_url.replace("https://", "").replace("/", "_")

def check_if_folder_exists(bucket_name, gcs_folder):
    """Checks if a folder exists in GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=f"{gcs_folder}/"))
    return len(blobs) > 0

def get_gcs_image_links(bucket_name, gcs_folder):
    """Fetches all image links from GCS for a specific book."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    prefix = f"{gcs_folder}/"
    
    blobs = list(bucket.list_blobs(prefix=prefix))
    image_links = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
    image_links.sort()
    
    print(f"✅ Retrieved {len(image_links)} images from GCS folder: {gcs_folder}")
    return image_links

def download_pdf_from_url(url, output_path=None):
    """
    If the URL contains a direct PDF, it downloads the file.
    Otherwise, it converts the webpage to a PDF using WeasyPrint.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type:
            # Handle PDF download
            if output_path is None:
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or 'downloaded_document.pdf'
                output_path = filename if filename.lower().endswith('.pdf') else filename + '.pdf'
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Successfully downloaded PDF to {output_path}")
            return output_path
        else:
            # Convert webpage to PDF using WeasyPrint
            if output_path is None:
                output_path = "converted_document.pdf"
            
            print(f"🚀 Converting webpage to PDF: {url}")
            HTML(url).write_pdf(output_path)
            print(f"✅ Successfully converted webpage to PDF: {output_path}")
            return output_path
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def upload_to_gcs(bucket_name, image, destination_blob_name, dataset):
    """Uploads an in-memory image to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Save image temporarily in current working directory as JPEG
    temp_image_path = os.path.join(os.getcwd(), destination_blob_name.replace("/", "_"))
    image.save(temp_image_path, "JPEG")
    
    # Upload to GCS
    blob.upload_from_filename(temp_image_path)
    gcs_link = f"gs://{bucket_name}/{destination_blob_name}"
    dataset.append(gcs_link)
    print(f"✅ Uploaded: {gcs_link}")
    
    # Remove temporary file
    os.remove(temp_image_path)
    
    return gcs_link

def load_pdf_and_upload(file_path, bucket_name, gcs_folder):
    """Loads each PDF page as an image, uploads it to GCS, and stores links in dataset."""
    start_time = time.time()
    dataset = []
    
    # Convert PDF pages to images
    images = convert_from_path(file_path)
    total_pages = len(images)
    
    for i, image in enumerate(tqdm(images, desc=f"Uploading {gcs_folder}", unit="page"), 1):
        destination_blob_name = f"{gcs_folder}/page_{i}.jpg"
        upload_to_gcs(bucket_name, image, destination_blob_name, dataset)
        
        elapsed_time = time.time() - start_time
        remaining_pages = total_pages - i
        print(f"Page {i}/{total_pages} - Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_pages}")
    
    return dataset

def download_gcs_image(gcs_uri):
    """Download an image from GCS and return it as a PIL Image."""
    try:
        if not gcs_uri.startswith("gs://"):
            raise ValueError("Invalid GCS URI format")
        
        storage_client = storage.Client()
        bucket_name = gcs_uri.split("/")[2]
        blob_path = "/".join(gcs_uri.split("/")[3:])
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        img_bytes = blob.download_as_bytes()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        return image
    
    except Exception as e:
        print(f"❌ Error downloading {gcs_uri}: {e}")
        return None

def download_gcs_image_bytes(gcs_uri):
    """Download an image from GCS and return its raw bytes."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI format")
    
    storage_client = storage.Client()
    bucket_name = gcs_uri.split("/")[2]
    blob_path = "/".join(gcs_uri.split("/")[3:])
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    return blob.download_as_bytes()

def get_patches(image_size, model_processor, model):
    """Get image patches for model processing"""
    return model_processor.get_n_patches(image_size,
                                         patch_size=model.patch_size,
                                         spatial_merge_size=model.spatial_merge_size)

def embed_and_mean_pool_batch(image_batch, model_processor, model, model_name):
    """Embed and mean pool a batch of images"""
    # Embed images
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device)
        image_embeddings = model(**processed_images)
    
    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()
    
    # Mean pooling
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []
    
    for image_embedding, tokenized_image, image in zip(image_embeddings,
                                                       processed_images.input_ids,
                                                       image_batch):
        # Handle NumPy array size
        image_size = image.shape[:2] if isinstance(image, np.ndarray) else image.size
        
        x_patches, y_patches = get_patches(image_size, model_processor, model)
        
        image_tokens_mask = (tokenized_image == model_processor.image_token_id)
        
        image_tokens = image_embedding[image_tokens_mask].view(x_patches, y_patches, model.dim)
        pooled_by_rows = torch.mean(image_tokens, dim=0)
        pooled_by_columns = torch.mean(image_tokens, dim=1)
        
        image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
        first_image_token_idx = image_token_idxs[0].cpu().item()
        last_image_token_idx = image_token_idxs[-1].cpu().item()
        
        prefix_tokens = image_embedding[:first_image_token_idx]
        postfix_tokens = image_embedding[last_image_token_idx + 1:]
        
        pooled_by_rows = torch.cat((prefix_tokens, pooled_by_rows, postfix_tokens), dim=0).cpu().float().numpy().tolist()
        pooled_by_columns = torch.cat((prefix_tokens, pooled_by_columns, postfix_tokens), dim=0).cpu().float().numpy().tolist()
        
        pooled_by_rows_batch.append(pooled_by_rows)
        pooled_by_columns_batch.append(pooled_by_columns)
    
    return image_embeddings_batch, pooled_by_rows_batch, pooled_by_columns_batch

def upload_batch(original_batch, pooled_by_rows_batch, pooled_by_columns_batch, payload_batch, collection_name, qdrant_client):
    """Upload a batch of embeddings to Qdrant"""
    try:
        qdrant_client.upload_collection(
            collection_name=collection_name,
            vectors={
                "mean_pooling_columns": pooled_by_columns_batch,
                "original": original_batch,
                "mean_pooling_rows": pooled_by_rows_batch
            },
            payload=payload_batch,
            ids=[str(uuid.uuid4()) for i in range(len(original_batch))]
        )
    except Exception as e:
        print(f"Error during upsert: {e}")

def batch_embed_query(query_batch, model_processor, model):
    """Embed a batch of queries"""
    with torch.no_grad():
        processed_queries = model_processor.process_queries(query_batch).to(model.device)
        query_embeddings_batch = model(**processed_queries)
    return query_embeddings_batch.cpu().float().numpy()

def reranking_search_batch(query_batch, collection_name, FilterList, qdrant_client, search_limit=20, prefetch_limit=200):
    """Search Qdrant with re-ranking"""
    filter_ = None
    
    if FilterList is not None:
        if not isinstance(FilterList, list):
            FilterList = [FilterList]
    
    # Apply filter only if FilterList is not empty
    if FilterList:
        filter_ = models.Filter(
            should=[
                models.FieldCondition(
                    key="url",
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
    
    response = qdrant_client.query_batch_points(
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
        response = qdrant_client.query_batch_points(
            collection_name=collection_name,
            requests=search_queries
        )
    
    return response

def process_search_query(query, content_url, model, processor, qdrant_client):
    """Process search query and get response from Gemini"""
    if isinstance(query, str):
        query = [query]
    
    colqwen_query = batch_embed_query(query, processor, model)
    answer_colqwen = reranking_search_batch(colqwen_query, COLLECTION_NAME, content_url, qdrant_client)
    
    top_10_results = []
    for point in answer_colqwen[0].points[:10]:
        top_10_results.append({"image_link": point.payload['image_link']})
    
    # Convert GCS images to inline data format
    images_payload = [
        types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=download_gcs_image_bytes(result["image_link"])
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
        client = genai.Client()
        print("Sending data to Gemini Vision API...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt_text)] + images_payload)]
        )
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return f"Error processing your query: {str(e)}"

def process_pdf(content_url, model, processor, qdrant_client):
    """Main function to process a PDF URL"""
    # Convert URL to GCS folder name
    gcs_folder = url_to_gcs_folder(content_url)
    
    # Check if folder already exists in GCS
    value = check_if_folder_exists(GCS_BUCKET_NAME, gcs_folder)
    
    if value:
        print("✅ Folder already exists in GCS. Fetching image links...")
        dataset = get_gcs_image_links(GCS_BUCKET_NAME, gcs_folder)
    else:
        print("🚀 Folder does not exist in GCS. Downloading PDF and uploading pages as images...")
        # Download the PDF
        pdf_file_path = download_pdf_from_url(content_url, "./output.pdf")
        if not pdf_file_path:
            return None, "Failed to download PDF"
        
        # Upload PDF pages to GCS
        dataset = load_pdf_and_upload(pdf_file_path, GCS_BUCKET_NAME, gcs_folder)
    
        # Process the pages and upload to Qdrant
        batch_size = 1  # Process 1 image at a time
        with tqdm(total=len(dataset), desc="Uploading to Qdrant") as pbar:
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i: i + batch_size]  # Select batch (list of GCS URIs)
                image_batch = []  # Store actual images
                
                # Download images from GCS
                for gcs_uri in batch:
                    image = download_gcs_image(gcs_uri)  # Keep as PIL Image
                    if image is not None:
                        image_batch.append(image)
                
                if not image_batch:  # If all images failed to download, skip this batch
                    continue
                
                current_batch_size = len(image_batch)
                
                original_batch, pooled_by_rows_batch, pooled_by_columns_batch = embed_and_mean_pool_batch(
                    image_batch, processor, model, MODEL_NAME
                )
                
                # Store image links directly
                payload_batch = [{"url": content_url, "image_link": gcs_uri} for gcs_uri in batch]
                
                try:
                    upload_batch(
                        np.asarray(original_batch, dtype=np.float32),
                        np.asarray(pooled_by_rows_batch, dtype=np.float32),
                        np.asarray(pooled_by_columns_batch, dtype=np.float32),
                        payload_batch,
                        COLLECTION_NAME,
                        qdrant_client
                    )
                except Exception as e:
                    print(f"Error during upsert: {e}")
                    continue
                
                pbar.update(current_batch_size)  # Update progress bar
        
        print("✅ Uploading complete!")
    return dataset, None

def create_langchain_graph(tools, llm_with_tools):
    """Create the LangChain LLM graph"""
    # System message
    sys_msg = SystemMessage(content=
                    """ - You are a Helpful AI Agent
                        - If you obtain a response from other llm then dont do anything, just check it relates to the user query and print the response to the user.
                        - Avoid unrelated or speculative information.
                        - Ensure the response is clear, concise, and directly addresses the user's query""")
    
    def assistant(state: MessagesState):
        return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}
    
    # Graph
    builder = StateGraph(MessagesState)
    
    # Define nodes
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.set_entry_point("assistant")
    
    builder.add_conditional_edges(
        "assistant", tools_condition
    )
    builder.add_edge("tools", "assistant")
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph

def process(content_url, user_query, google_api_key, google_credentials_path):
    """Main entry point for the application"""
    # Set environment variables
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
    
    # Initialize clients and models
    model, processor = initialize_model()
    qdrant_client = initialize_qdrant_client()
    genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    # Process PDF
    dataset, error = process_pdf(content_url, model, processor, qdrant_client)
    if error:
        return f"Error: {error}"
    
    # Create search tool
    @tool
    def search_pdf(query: str) -> str:
        """
        Searches the PDF for information related to the query.
        
        Args:
            query: The search query
            
        Returns:
            Relevant information from the PDF
        """
        return process_search_query(query, content_url, model, processor, qdrant_client)
    
    # Now use this tool
    tools = [search_pdf]
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Create LangChain graph
    graph = create_langchain_graph(tools, llm_with_tools)
    
    # Process user query
    thread_id = {"configurable": {"thread_id": "1"}}
    input_message = [HumanMessage(content=user_query)]
    
    # Use invoke instead of __call__
    msg = graph.invoke({"messages": input_message}, thread_id)
    
    # Return the result
    return msg['messages'][-1].content

if __name__ == "__main__":
    # Example usage
    content_url = "https://www.w3schools.com/python/python_intro.asp"
    user_query = "Why python and what can it do?"
    google_api_key = "AIzaSyA9I36rGn5QFV-GP8YyUvmBAzQaOsyELi4"
    google_credentials_path = "/Users/saivignesh/Documents/DGM_Project/Browser-Extension/fabled-emblem-450414-r0-a631ebb8abd9.json"
    
    result = process(content_url, user_query, google_api_key, google_credentials_path)
    print(result)