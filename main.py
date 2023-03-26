import pandas as pd
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Union


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
skincare=pd.read_csv("skincare_original.csv")

searchList=pd.concat([makeup[["id","brand","name","img"]],skincare[["id","brand","name","img"]]])
#searchList=searchList.set_index(["id"])
# mergeList=pd.concat(df,df2)
# searchList=mergeList[["name","img"]]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/search/{q}",response_class=ORJSONResponse)
async def search(q: Union[str, None] = None):
      result=None
      if q=="all_products":
           result=searchList
      else:
          resultBrand=searchList[searchList["brand"].str.contains(q)]
          resultName = searchList[searchList["name"].str.contains(q)]

          result=pd.concat([resultBrand,resultName])

      result=result.drop_duplicates()
      return result.to_json(orient="records")