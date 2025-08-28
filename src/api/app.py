from fastapi import FastAPI, APIRouter, Form
from fastapi.responses import FileResponse, JSONResponse
from uuid import UUID
import os

from generator import Word2Vec, SpectrogramGenerator, Vocoder


class App:
    DATASTORE_PATH = os.path.join(os.getcwd(), "store")

    def __init__(self):
        self.__app = FastAPI()
        self.__router = APIRouter()
        self.__word2vec = Word2Vec()
        self.__spec_generator = SpectrogramGenerator()
        self.__vocoder = Vocoder()

        self.__setup_routes()

    def __setup_routes(self): ...

    def get_app(self) -> FastAPI:
        self.__app.include_router(self.__router)
        return self.__app

    async def generate_audio(self, word: str = Form(...)) -> JSONResponse: ...
