import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection setup
mongo_uri = os.getenv("MONGO_CONNECTION_STRING")
db_name = os.getenv("MONGO_DB_NAME")
collection_name = os.getenv("MONGO_COLLECTION_NAME")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# Delete all existing documents
result = collection.delete_many({})
print(f"Deleted {result.deleted_count} documents from the collection.")
