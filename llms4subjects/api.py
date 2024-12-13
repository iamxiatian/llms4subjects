import requests

EMBEDDING_SERVER_URL = "http://10.96.1.43:8188/emb"
EMBEDDING_DIM = 1024

def get_embedding(text: str) -> list[float]:
    """生成embedding的接口，通过调用RESTful API实现"""
    headers = {
        "Content-Type": "application/json",
    }
    data = {"text": [text]}
    response = requests.post(EMBEDDING_SERVER_URL, headers=headers, json=data)
    response = response.json()
    embedding = response["result"][0]
    return embedding
