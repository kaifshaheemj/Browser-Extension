import os
import time
import uuid
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from google.cloud import storage

# Configure Qdrant client
qdrantclient = QdrantClient(
    url="https://3bed5a5a-f523-4ae1-9ed0-d5b4408b8449.eu-west-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.oC3xUAMljhPccXJQEZX1LjbXuO3CqYJ8CUwFZI2aGHk",
    timeout=100
)
collection_name = "dgmproject"

# Check if collection exists
try:
    qdrantclient.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e:
    if "Collection not found" in str(e):
        print(f"Collection '{collection_name}' not found")
        # We'll create the collection later in the code


# For local use, you'll need to download your credential file and set its path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/saivignesh/Documents/DGM_Project/Browser-Extension/fabled-emblem-450414-r0-a631ebb8abd9.json"

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Initialize ColQwen2 model
model_name = "vidore/colqwen2-v0.1"

device = "cpu"  # Use MPS for Apple Silicon, CPU otherwise
print(f"Using device: {device}")    

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,  # Use float32 if on CPU
    device_map=device,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Configuration
Book_Name = "geeksforgeeks"  # Change as needed
Book_Category = ["geeksforgeeks"]  # Change as needed
gcs_bucket_name = "dgmtest"  # Your GCS bucket name

# Create a temporary directory for intermediate files
temp_dir = os.path.join(os.getcwd(), "temp_images")
os.makedirs(temp_dir, exist_ok=True)

def upload_to_gcs(bucket_name, image, destination_blob_name):
    """Uploads an image to Google Cloud Storage and returns the GCS link."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Save image temporarily
        temp_image_path = os.path.join(temp_dir, os.path.basename(destination_blob_name))
        image.save(temp_image_path, "JPEG")

        # Upload to GCS
        blob.upload_from_filename(temp_image_path)
        gcs_link = f"gs://{bucket_name}/{destination_blob_name}"
        print(f"✅ Uploaded: {gcs_link}")

        # Remove temp file
        os.remove(temp_image_path)
        
        return gcs_link
    except Exception as e:
        print(f"❌ Error uploading to GCS: {e}")
        return None

def download_gcs_image(gcs_uri):
    """Download an image from GCS and return it as a PIL Image."""
    try:
        # Extract bucket and blob path
        if not gcs_uri.startswith("gs://"):
            raise ValueError("Invalid GCS URI format")

        bucket_name = gcs_uri.split("/")[2]  # Extract bucket name correctly
        blob_path = "/".join(gcs_uri.split("/")[3:])  # Extract blob path correctly

        # Get GCS bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download image to memory
        img_bytes = blob.download_as_bytes()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")  # Keep it as a PIL image
        return image

    except Exception as e:
        print(f"❌ Error downloading {gcs_uri}: {e}")
        return None  # Return None if download fails

def get_gcs_image_links(bucket_name, book_name, gcs_folder="pdf_images"):
    """Fetches all image links from GCS for a specific book."""
    try:
        bucket = storage_client.bucket(bucket_name)
        prefix = f"{gcs_folder}/{book_name}/"  # Folder path

        # List all blobs (files) in the specified folder
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Extract GCS links
        image_links = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]

        # Sort to maintain correct page order
        image_links.sort()

        print(f"✅ Retrieved {len(image_links)} image links for {book_name}")
        return image_links
    except Exception as e:
        print(f"❌ Error fetching GCS image links: {e}")
        return []

def load_pdf_and_upload(file_path, bucket_name, book_name, gcs_folder="pdf_images", batch_size=100):
    """Loads PDF pages in batches, uploads them to GCS, and returns the list of GCS links."""
    start_time = time.time()
    dataset = []  # Store GCS links
    
    # Determine total pages
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        total_pages = len(reader.pages)
    
    page_number = 1
    
    while page_number <= total_pages:
        print(f"\n🚀 Processing pages {page_number} to {min(page_number + batch_size - 1, total_pages)}...")
        
        # Load pages in batch
        images = convert_from_path(
            file_path, 
            first_page=page_number, 
            last_page=min(page_number + batch_size - 1, total_pages)
        )
        
        for i, image in enumerate(tqdm(images, desc="Uploading to GCS", unit="page"), 0):
            current_page = page_number + i
            destination_blob_name = f"{gcs_folder}/{book_name}/page_{current_page}.jpg"
            
            # Upload image to GCS
            gcs_link = upload_to_gcs(bucket_name, image, destination_blob_name)
            if gcs_link:
                dataset.append(gcs_link)
            
            elapsed_time = time.time() - start_time
            print(f"Page {current_page}/{total_pages} - Elapsed: {elapsed_time:.2f}s")
        
        page_number += len(images)
        print(f"✅ Completed batch: {max(1, page_number - len(images))} to {page_number - 1}. Freeing memory...\n")
        
        # Free memory before loading next batch
        del images
        time.sleep(1)
    
    return dataset

# Function to create Qdrant collection if it doesn't exist
def create_qdrant_collection():
    try:
        qdrantclient.create_collection(
            collection_name=collection_name,
            vectors_config={
                "original": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=0  # Switching off HNSW
                    ),
                    on_disk=True  # Store vectors on disk
                ),
                "mean_pooling_columns": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    on_disk=True  # Store vectors on disk
                ),
                "mean_pooling_rows": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    on_disk=True  # Store vectors on disk
                )
            }
        )

        qdrantclient.create_payload_index(
            collection_name=collection_name,
            field_name="Book_Category",
            field_schema=models.PayloadSchemaType.KEYWORD)
        
        print(f"✅ Collection '{collection_name}' created successfully")
    except Exception as e:
        print(f"Error creating collection: {e}")

# Functions for embedding and storing to Qdrant
def get_patches(image_size, model_processor, model):
    return model_processor.get_n_patches(image_size,
                                         patch_size=model.patch_size,
                                         spatial_merge_size=model.spatial_merge_size)

def embed_and_mean_pool_batch(image_batch, model_processor, model, model_name):
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch)

    print("Processed Images Type:", type(processed_images))
    print(f"Processed Images Keys: {processed_images.keys()}")

    # Ensure "pixel_values" exists and is a tensor
    if all(key in processed_images for key in ["pixel_values", "image_grid_thw", "input_ids", "attention_mask"]): 
        pixel_values = processed_images["pixel_values"]
        image_grid_thw = processed_images["image_grid_thw"]
        input_ids = processed_images["input_ids"]
        attention_mask = processed_images["attention_mask"]  # Extract attention_mask

        print("Processed Images Shape:", pixel_values.shape if isinstance(pixel_values, torch.Tensor) else "Not a Tensor")
        
        # Convert to correct data types
        pixel_values = pixel_values.to(torch.float32).to("cpu")
        image_grid_thw = image_grid_thw.to(torch.int64).to("cpu")  
        input_ids = input_ids.to(torch.int64).to("cpu")  
        attention_mask = attention_mask.to(torch.int64).to("cpu")  # Ensure attention_mask is an integer tensor

        print(f"Processed Images Shape After Conversion: {pixel_values.shape}")
        print(f"image_grid_thw Shape: {image_grid_thw.shape}")
        print(f"input_ids Shape: {input_ids.shape}")
        print(f"attention_mask Shape: {attention_mask.shape}")

        # Pass all required inputs to the model
        image_embeddings = model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask  # Add this!
        )

    else:
        print("Error: Required keys ('pixel_values', 'image_grid_thw', 'input_ids', 'attention_mask') not found in processed_images")

    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()


    # Mean pooling
    pooled_by_rows_batch = []
    pooled_by_columns_batch = []

    for image_embedding, tokenized_image, image in zip(image_embeddings,
                                                     processed_images.input_ids,
                                                     image_batch):
        # Handle PIL Image size
        image_size = image.size

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

def upload_batch(original_batch, pooled_by_rows_batch, pooled_by_columns_batch, payload_batch, collection_name):
    try:
        qdrantclient.upload_collection(
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

# Main execution function
def process_pdf(pdf_file_path):
    # Step 1: Create the collection if it doesn't exist
    create_qdrant_collection()
    
    # Step 2: Convert PDF and upload images to GCS
    print(f"Converting PDF and uploading images to GCS: {pdf_file_path}")
    dataset = load_pdf_and_upload(pdf_file_path, gcs_bucket_name, Book_Name)
    
    # Alternatively, if images are already uploaded, you can retrieve them:
    # dataset = get_gcs_image_links(gcs_bucket_name, Book_Name)
    
    if not dataset:
        print("No images found. Exiting...")
        return
        
    # Step 3: Process images and upload to Qdrant
    batch_size = 1  # Process 1 image at a time
    
    with tqdm(total=len(dataset), desc="Embedding and Uploading") as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]  # Select batch of GCS URIs
            image_batch = []  # Store actual images
            
            # Download images from GCS
            for gcs_uri in batch:
                image = download_gcs_image(gcs_uri)
                if image is not None:
                    image_batch.append(image)
            
            if not image_batch:  # If all images failed to download, skip this batch
                continue
                
            current_batch_size = len(image_batch)
            
            original_batch, pooled_by_rows_batch, pooled_by_columns_batch = embed_and_mean_pool_batch(
                image_batch, processor, model, model_name
            )
            
            # Store image links in payload
            payload_batch = [{"Book_Category": Book_Category, "Book_Name": Book_Name, "image_link": gcs_uri} for gcs_uri in batch]
            
            try:
                upload_batch(
                    np.asarray(original_batch, dtype=np.float32),
                    np.asarray(pooled_by_rows_batch, dtype=np.float32),
                    np.asarray(pooled_by_columns_batch, dtype=np.float32),
                    payload_batch,
                    collection_name
                )
            except Exception as e:
                print(f"Error during upsert: {e}")
                continue
                
            pbar.update(current_batch_size)  # Update progress bar
    
    # Clean up temp directory
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
        
    print("✅ Processing complete!")

# To run the code:
if __name__ == "__main__":
    # Set your PDF file path here
    pdf_file_path = "/Users/saivignesh/Documents/DGM_Project/Browser-Extension/output.pdf"  # Update with your actual PDF path
    process_pdf(pdf_file_path)