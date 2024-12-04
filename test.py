import os
from pymongo import MongoClient
from pymongo.errors import PyMongoError

def test_mongo_connection():
    # MongoDB connection details
    mongo_connection_string = "mongodb+srv://cazton:WelcomeTerminators1@caztoncosmonaut.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
    mongo_db_name = "caztoncosmos"
    mongo_collection_name = "doc2"

    try:
        # Create a MongoClient to the running mongod instance
        client = MongoClient(mongo_connection_string)

        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB.")

        # Access the specified database and collection
        db = client[mongo_db_name]
        collection = db[mongo_collection_name]

        # Check if there are any documents in the collection
        document_count = collection.count_documents({})
        if document_count > 0:
            print(f"The collection '{mongo_collection_name}' in database '{mongo_db_name}' contains {document_count} documents.")
        else:
            print(f"The collection '{mongo_collection_name}' in database '{mongo_db_name}' is empty.")
    except PyMongoError as e:
        print(f"Failed to connect to MongoDB or query the collection: {e}")

if __name__ == "__main__":
    test_mongo_connection()