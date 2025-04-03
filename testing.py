
import torch, uuid, base64, os, time, json, ast
from io import BytesIO
import google.generativeai as genai
import openai
import pdfkit
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from google.cloud import storage
from google import genai
from google.genai import types
from PIL import Image
from IPython.display import display
import numpy as np
from tqdm import tqdm
from model import model, processor



'''
config = pdfkit.configuration(wkhtmltopdf='/usr/bin/wkhtmltopdf')
url = r"https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/"
pdfkit.from_url(url, "/content/output.pdf", configuration=config)
'''


"""## PDF into Images ##"""

Book_Name = "geeksforgeeks"
Book_Category = ["geeksforgeeks"]



"""##Querying"""

query = ["Give me the code for DBSCAN Implementation"]
query[0]

#query
if not isinstance(query,list):
    query = [query]

def batch_embed_query(query_batch, model_processor, model):
    with torch.no_grad():
        processed_queries = model_processor.process_queries(query_batch).to(model.device)
        query_embeddings_batch = model(**processed_queries)
    return query_embeddings_batch.cpu().float().numpy()

colqwen_query = batch_embed_query(query, processor, model)

with torch.no_grad():
    batch_query = processor.process_queries(query).to(
        model.device
    )
    print(processor.tokenizer.tokenize(
        processor.decode(batch_query.input_ids[0])
    ))


# Initialize Google Generative AI Client
client = genai.Client(api_key=api_key)

def LLM_filter(query):
    """Categorizes the user query into any of the predefined Book_Categories, it can also be list of categories"""

    Book_Categories = ["geeksforgeeks"]

    print("Categorizing user query...")

    # Construct the prompt for categorization
    prompt_text = f"""
    You are an expert in medical literature categorization.
    Categorize the following query into one of the predefined Book_Categories: {', '.join(Book_Categories)}.

    User's Query:
    {query}

    Your task:
    - Identify the most relevant category based on the content of the query.
    - Return only the category names as a List. If uncertain, return empty list.
    - No preambles,No extra text, explanations, or formatting.
    """

    # Send request to Gemini Vision API
    try:
        print("Sending query to Gemini Vision API for categorization...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt_text)])]
        )

        print("API Response:", response.text)
        category_list = ast.literal_eval(response.text.strip())

        if isinstance(category_list, list):
            print("Categorized Response:", category_list)
            return category_list
        else:
            print("Invalid format received, returning empty list.")
            return []

    except Exception as e:
        print(f"Error occurred: {e}")
        return "General"

# Ensure the user query is properly formatted
LLM_category = LLM_filter(query)

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
answer_colqwen[0].points

from IPython.display import display
from PIL import Image

top_10_results = []
for point in answer_colqwen[0].points[:10]:
  top_10_results.append({"image_link": point.payload['image_link']})
top_10_results

# Ensure the user query is properly formatted
user_query = query + ["Give me in detail explanation"]
response_text = LLM_with_images(top_10_results, user_query)


'''
#for showing the top k images

# Initialize Google Cloud Storage client
storage_client = storage.Client()

def download_gcs_image(gcs_uri):
    """Download an image from GCS and return it as a PIL Image."""
    assert gcs_uri.startswith("gs://"), "Invalid GCS URI format"

    bucket_name = gcs_uri.split("/")[2]  # Extract bucket name
    blob_path = "/".join(gcs_uri.split("/")[3:])  # Extract file path

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    img_bytes = blob.download_as_bytes()
    return Image.open(BytesIO(img_bytes)).convert("RGB")  # Return a PIL Image

# Process and display images
for result in top_10_results:
    image = download_gcs_image(result['image_link'])  # Download image from GCS
    display(image)

'''
