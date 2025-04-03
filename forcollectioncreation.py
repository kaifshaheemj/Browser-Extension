#for creating collection in qdrant database (below are not default configurations, it is set according to our necessity, particulary for colpali and colqwen)
##FIXED FROM ADMIN SIDE (ONE TIME COLLECTION CREATION for this usecase)

from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models


# Step 1: Initialize the Qdrant client
qdrantclient = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
    timeout=100
)
collection_name=os.getenv('QDRANT_COLLECTION_NAME')

#collection configuration according to the usecase
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

#create a payload index for retrivel filter search
qdrantclient.create_payload_index(
    collection_name=collection_name,
    field_name="url",
    field_schema=models.PayloadSchemaType.KEYWORD)

# Step 3: Verify the collection creation
collections = qdrantclient.get_collections()
print("Collections:", collections)