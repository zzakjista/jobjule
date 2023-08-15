
from fastapi import FastAPI, Depends, Path, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from fastapi.responses import StreamingResponse , HTMLResponse
import io
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from crud import CRUD


templates = Jinja2Templates(directory="C:/github/worknet_recommender/app/templates") #로컬에서 실행 시 절대 경로를 수정해주세요

app = FastAPI(title="worknet", description="worknet", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# sql 서버 구동 시 실행해주세요.
# engine = engineconn()
# session = engine.sessionmaker()



@app.get("/", response_class=HTMLResponse)
async def letswork(request: Request):
    return templates.TemplateResponse("letswork.html", context={'request': request})  

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request, page: int=1):
    crud = CRUD()
    total_page, list = crud.get_page(page)
    list = list[['구인인증번호','회사명','구인제목','근무지역','최소임금액']].to_dict('records')
    return templates.TemplateResponse("home.html", context={'request': request, "list":list, "page":page})

@app.get("/detail", response_class=HTMLResponse)
async def recommender(request: Request, id : str):
    crud = CRUD()
    _, data = crud.get_item()
    recommendation_result = crud.get_recommend_item(id)
    data = data[data['구인인증번호'].isin(recommendation_result)]
    detail_dict = {}
    for key in recommendation_result:
        detail_dict[key] = data[data['구인인증번호']==key].to_dict('records')
    return templates.TemplateResponse("detail.html", context={'request': request, "data":detail_dict, 'recommendation_result':recommendation_result, 'id':id})
