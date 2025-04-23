
from datasets import load_dataset

import pandas as pd

#dataset=load_dataset("AIatMongoDB/embedded_movies")
dataset=load_dataset("MongoDB/embedded_movies")
#MongoDB/embedded_movies

dataset

dataset_df=pd.DataFrame(dataset["train"])

dataset_df.head()

dataset_df.columns

dataset_df["plot"][0]

dataset_df["fullplot"][0]

dataset_df["num_mflix_comments"][0]

dataset_df["fullplot"].isnull().sum()

dataset_df.shape

dataset_df["poster"][0]

dataset_df.isnull().sum()

dataset_df=dataset_df.dropna(subset=["fullplot"])

dataset_df["fullplot"].isnull().sum()

dataset_df = dataset_df.drop(columns=["plot_embedding"])

dataset_df.head(2)

# @title metacritic

from matplotlib import pyplot as plt
dataset_df['metacritic'].plot(kind='hist', bins=20, title='metacritic')
plt.gca().spines[['top', 'right',]].set_visible(False)

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("thenlper/gte-large")

dataset_df["fullplot"][2]

text="   sunny savita is  a data scientist who create prodcut of data"

text="   sunny savita is  a data scientist who create prodcut of data     "

text

text.strip()

def get_embedding(text:str)->list[float]:

  if not text.strip():
    print("attempted to get embedding for empty text.")
    return []

  embedding=embedding_model.encode(text)
  return embedding.tolist()

dataset_df["embedding"]=dataset_df["fullplot"].apply(get_embedding)

dataset_df.head(3)

import pymongo

#!python -m pip install "pymongo[srv]"

from pymongo.mongo_client import MongoClient

from google.colab import userdata
uri=userdata.get('MONGO_URI')

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

def get_mongo_client(uri):
  try:
    client = MongoClient(uri)
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    return client
  except Exception as e:
    print(e)
    return None

mongo_client=get_mongo_client(uri)

db=mongo_client["moviedb2"]

collection=db["moviecollection2"]

collection.insert_one({"name":"sunny",
                       "designation": "genai engineer",
                       "location":"bangaluru",
                       "mailid":"sunny.savita@ineuron.ai"})

collection.insert_one({"name":"dipesh",
                       "designation": "ops manager",
                       "location":"bangaluru"})

collection2=db["moviecollectionsecond"]

collection2.insert_one({"name":"krish",
                       "designation": "tech lead",
                       "location":"bangaluru",
                        "phonenumber":57454745834})

collection.delete_many({})

dataset_df.tail(3)

document=dataset_df.to_dict("records")

collection.insert_many(document)

print("data ingestion in mongodb is completed")

"""# Data Retrival"""

{
    key:value
}

{
 "fields": [{
     "numDimensions": 1024,
     "path": "embedding",
     "similarity": "cosine",
     "type": "vector"
   }]
}

user_query="what is the best horror movie?"

query_embedding=get_embedding(user_query)

query_embedding

print(query_embedding)

"""https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/

"""

pipeline = [

    {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 4,  # Return top 4 matches
        }
    },
    {
        "$project": {
            "fullplot": 1,  # Include the plot field
            "title": 1,  # Include the title field
            "genres": 1,  # Include the genres field
            "score": {"$meta": "vectorSearchScore"},  # Include the search score
        }
    }
]

collection.aggregate(pipeline)

list(collection.aggregate(pipeline))

def get_embedding(text:str)->list[float]:

  if not text.strip():
    print("attempted to get embedding for empty text.")
    return []

  embedding=embedding_model.encode(text)
  return embedding.tolist()

def vector_search(user_query,collection):

  query_embedding=get_embedding(user_query)
  print(query_embedding)

  if query_embedding is None:
    return "Invalid query or embeddig is failed"

  pipeline=[

            {
                "$vectorSearch":{

                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 4,  # Return top 4 matches


                }

            },

              {
                 "$project":{

                "fullplot": 1,  # Include the plot field
                "title": 1,  # Include the title field
                "genres": 1,  # Include the genres field
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
                 }

            }

           ]

  result=collection.aggregate(pipeline)
  return list(result)

vector_search("what is the best horror movie to watch and why?",collection)

def get_search_result(query,collection):

  get_knowledge=vector_search(query,collection)

  search_result=""

  for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"

  return search_result

query="what is the best comedy movie to watch and why?"

collection

source_information=get_search_result(query,collection)

source_information

combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."

print(combined_information)




from huggingface_hub import notebook_login
notebook_login()

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# CPU Enabled uncomment below 👇🏽
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
# GPU Enabled use below 👇🏽
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

# Moving tensors to GPU
input_ids = tokenizer(combined_information, return_tensors="pt").to("cuda")

response = model.generate(**input_ids, max_new_tokens=500)

print(tokenizer.decode(response[0]))

#https://python.langchain.com/docs/integrations/retrievers/weaviate-hybrid/


https://towardsdatascience.com/improving-retrieval-performance-in-rag-pipelines-with-hybrid-search-c75203c2f2f5
https://esteininger.medium.com/mongodb-and-pinecone-building-real-time-ai-applications-cd8e0482a3c7

# you are supposed to solve these two thing(hybrid search,combination of db(pinecone+mongodb)) you can send me this notebook

# i will upload these notebook in resource section with your name

# i will create one video which will be dedicated to that best solution and i will do linkedin post from my linkedin account and i wll mention that person as well.