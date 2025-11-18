from fastapi import FastAPI

import requests

if __name__ == "__main__":
    response = requests.post(
        "http://localhost:8000/query",
        json={"query": "客户经理的考核标准是什么?"}
    )

    result = response.json()
    print(result["answer"])
