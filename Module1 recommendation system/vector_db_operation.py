from pinecone import Pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer


def vectorize_and_upload(api_key, index_name, csv_file, embedding_model_name="all-mpnet-base-v2"):
    pc = Pinecone(api_key)
    index = pc.Index(index_name)

    # Load the sentence transformer model
    model = SentenceTransformer(embedding_model_name)

    # Read the CSV data
    data = pd.read_csv(csv_file)

    # Preprocess the description text (optional)
    # You can add any preprocessing steps like tokenization, stop word removal etc. here
    data["Description"] = data["Description"].str.lower()

    # Generate embeddings for the descriptions

    embeddings = model.encode(data["Description"].tolist())

    # Create a list of dictionaries for Pinecone upload
    data_for_upload = []
    for i, row in data.iterrows():
        # Convert numpy array to list and wrap it in a dictionary under 'values' key
        vector_data = {'values': embeddings[i].tolist()}

        data_for_upload.append({
            "id": str(row["id"]),  # Assuming you have a unique 'id' column
            "vector": vector_data,
            "metadata": {
                "description": row["Description"],
                "customer_id": row["CustomerID"],  # Include additional fields here
                "country": row["Country"],
                # Add other relevant features from your columns (optional)
                # "price": row["Price"],  # Example
                # "brand": row["Brand"],   # Example
            }
        })

    # Upload the data to Pinecone
    index.upsert(data_for_upload)

    print("Data uploaded successfully!")


# Replace with your actual API key and index name
vectorize_and_upload(
    api_key="a2d61230-ed5f-4887-9107-53c455b85b68", index_name="product-recommendation", csv_file="cleaned_data1.csv"
)
