from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch, uuid, os, json
import numpy as np
import dotenv

from model import model, processor

# Load environment variables
dotenv.load_dotenv()

# Fetch Qdrant credentials from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate credentials
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing from the environment variables.")

# Creating Qdrant Client
qdrantclient = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=100
)

# Collection name
collection_name = "dgm"

# Check if the collection exists
try:
    qdrantclient.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
    collection_exists = True
except Exception as e:
    if "Collection not found" in str(e):
        print(f"Collection '{collection_name}' not found. Creating a new collection...")
        collection_exists = False
    else:
        raise e  # Raise other exceptions

# Create the collection only if it does not exist
if not collection_exists:
    qdrantclient.create_collection(
        collection_name=collection_name,
        vectors_config={
            "original": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # Switching off HNSW
                on_disk=True
            ),
            "mean_pooling_columns": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                on_disk=True
            ),
            "mean_pooling_rows": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                on_disk=True
            )
        }
    )

# Creating a Payload Index
qdrantclient.create_payload_index(
        collection_name=collection_name,
        field_name="Book_Category",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

# Embeddings and Storing into Qdrant
def get_patches(image_size, model_processor, model):
    return model_processor.get_n_patches(
        image_size,
        patch_size=model.patch_size,
        spatial_merge_size=model.spatial_merge_size
    )

def embed_and_mean_pool_batch(image_batch, model_processor, model, model_name):
    with torch.no_grad():
        processed_images = model_processor.process_images(image_batch).to(model.device)
        image_embeddings = model(**processed_images)

    image_embeddings_batch = image_embeddings.cpu().float().numpy().tolist()
    pooled_by_rows_batch, pooled_by_columns_batch = [], []

    for image_embedding, tokenized_image, image in zip(image_embeddings, processed_images.input_ids, image_batch):
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

# Uploading Batch into Qdrant
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
            ids=[str(uuid.uuid4()) for _ in range(len(original_batch))]
        )
    except Exception as e:
        print(f"Error during upsert: {e}")


storage_client = storage.Client()

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


batch_size = 1  # Process 1 image at a time
dataset_source = dataset  # List of GCS image links

with tqdm(total=len(dataset), desc="Uploading") as pbar:
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
            image_batch, processor, model, model_name
        )

        # Store image links directly
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
            print(f" Error during upsert: {e}")
            continue

        pbar.update(current_batch_size)  # Update progress bar

print("✅ Uploading complete!")