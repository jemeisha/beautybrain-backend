import pandas as pd
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
makeup=pd.read_csv("makeup_final_updated.csv")
skincare=pd.read_csv("skincare_merged_final.csv")

searchList=pd.concat([makeup[["id","brand","name","img"]],skincare[["id","brand","name","img"]]])
#searchList=searchList.set_index(["id"])
# mergeList=pd.concat(df,df2)
# searchList=mergeList[["name","img"]]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/search/{q}",response_class=ORJSONResponse)
async def search(q: str):

      resultBrand=searchList[searchList["brand"].str.contains(q)]
      resultName = searchList[searchList["name"].str.contains(q)]

      result=pd.concat([resultBrand,resultName])
      #result=resultBrand
      result=result.drop_duplicates()

      #return ORJSONResponse(result.to_json())
      return resultName.to_json(orient="records")