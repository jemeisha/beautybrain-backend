import json

import pandas as pd
from fastapi import FastAPI, Request, Query
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
from recommender import recommend_products



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
makeup = pd.read_csv("makeup_final_updated.csv")
skincare = pd.read_csv("skincare_original.csv")

makeup["product_type"]="makeup"
skincare["product_type"]="skincare"

makeup["skin_type"]=makeup["skin_type"].str.lower()
skincare["skin_type"]=skincare["skin_type"].str.lower()

makeup["tags"] = makeup["label"] + ',' + makeup["formulation"] + ',' + makeup["skin_type"]
makeup["rating"] = makeup["rating"].astype(str)

skincare["tags"] = skincare["label"] + ',' + skincare["fomula"] + ',' + skincare["skin_type"]
skincare["rating"] = " "

makeup.fillna("",inplace=True)
skincare.fillna("",inplace=True)


mergedList= pd.concat([makeup,skincare])
searchList = pd.concat([makeup[["id", "brand", "name", "img", "tags", "rating","label","concern"]],
                        skincare[["id", "brand", "name", "img", "tags", "rating","label","concern","concern2","concern3"]]])


# searchList=searchList.set_index(["id"])
# mergeList=pd.concat(df,df2)
# searchList=mergeList[["name","img"]]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def root(request: Request):
    json_body = await request.json()
    # print(json_body)
    products,productsAns = recommend_products(
        json_body["answers"],
        json_body["imgData"],
        json_body["output"],
        makeup,
        skincare
    )
    #return searchList.head(100).to_json(orient="records")
    productJson=products.to_json(orient="records")
    answerJson=productsAns.to_json(orient="records")
    data={
        "recommendedProducts":json.loads(productJson),
        "answerBasedProducts":json.loads(answerJson)
    }
    return json.dumps(data)


@app.get("/search/{q}", response_class=ORJSONResponse)
async def search(
                 q: Union[str, None] = None,
                 category:str=Query(None),
                 concern:str=Query(None)
                  ):
    result = None
    print("Category: ",category)
    print("Concern: ",concern)

    allProducts=searchList
    if category != "all":
        allProducts=allProducts[allProducts["label"].str.contains(category, case=False)]

    if concern != "all":
        allProducts=allProducts[
            allProducts["concern"].str.contains(concern, case=False) |
            allProducts["concern2"].str.contains(concern, case=False) |
            allProducts["concern3"].str.contains(concern, case=False)
        ]

    if q == "all_products":
        result = allProducts
    else:
        resultBrand = allProducts[allProducts["brand"].str.contains(q)]
        resultName = allProducts[allProducts["name"].str.contains(q)]

        result = pd.concat([resultBrand, resultName])

    result = result.drop_duplicates()
    return result.to_json(orient="records")

@app.get("/product/{id}", response_class=ORJSONResponse)
async def product(
                 id: Union[str, None] = None
                  ):
    print("id: ", id)
    filtered=mergedList.loc[mergedList["id"]==id]
    return filtered.to_json(orient="records")
