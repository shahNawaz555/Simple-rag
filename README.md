🎬 MongoDB + HuggingFace RAG Pipeline with Vector Search
This notebook demonstrates a practical Retrieval-Augmented Generation (RAG) pipeline using:

🧠 Sentence embeddings with SentenceTransformers

📦 Data ingestion and retrieval with MongoDB

🤖 Answer generation using Gemma-2B-IT from HuggingFace

🔍 Future expansion with Hybrid Search combining Pinecone and MongoDB

🗂️ Dataset: embedded_movies
We use the MongoDB/embedded_movies dataset from Hugging Face which contains metadata for thousands of movies including:

title

genres

fullplot

metacritic

embedding (which we generate)

📊 Data Preparation
Loaded the dataset using datasets library

Cleaned NaN values from fullplot

Removed unused columns

Generated vector embeddings using thenlper/gte-large

🧬 Embedding Generation
python
Copy
Edit
def get_embedding(text:str) -> list[float]:
    if not text.strip():
        return []
    embedding = embedding_model.encode(text)
    return embedding.tolist()
Embeddings are computed for each movie's fullplot and added to MongoDB.

☁️ MongoDB Integration
Connected to MongoDB Atlas using pymongo

Inserted full documents along with vector embeddings

Verified insertion with manual queries

🔎 Vector Search with MongoDB Atlas
Used the $vectorSearch aggregation stage in MongoDB to:

Query based on semantic similarity of user input

Return top 4 most relevant matches by cosine similarity

🧠 RAG Generation using google/gemma-2b-it
Query embedding is used to retrieve top matching movie plots

Combined user query and search results into a single prompt

Prompt passed to Gemma-2B to generate a natural language answer
