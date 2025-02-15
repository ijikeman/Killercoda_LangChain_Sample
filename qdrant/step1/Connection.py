from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
import numpy as np

# Qdrant クライアント設定 (localhost:6333)
client = QdrantClient("localhost", port=6333)

# 埋め込みモデルのロード
model = SentenceTransformer("all-MiniLM-L6-v2")

# コレクションの作成
collection_name = "my_collection"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# テキストデータ
texts = ["This is a test.", "Hello world!", "Qdrant is a vector DB."]
embeddings = model.encode(texts).tolist() # モデルが 各テキストを数値ベクトルに変換 します。

# Qdrantにデータを格納
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"text": texts[i]})
    for i in range(len(texts))
]
client.upsert(collection_name=collection_name, points=points)

print("Data inserted successfully.")
