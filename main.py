import pandas as pd
from fastapi import FastAPI

app = FastAPI()
df=pd.read_csv("makeup_final_updated.csv")

searchlist=df[["name","img"]]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/search/{q}")
async def search(q: str):
      result=searchlist[searchlist["name"].str.contains(q)]

      return result.to_json()
    # return searchlist.to_json()