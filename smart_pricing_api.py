from fastapi import FastAPI, Depends, Response, status, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from endpoint_classes import ServicesPayload
from pydantic import ValidationError
from reverse_auction_env import ReverseAuctionEnv
import os
import cv2
import glob
from train_and_evaluation_functions import evaluate,create_video_from_pngs



smart_pricing_api = FastAPI(
    title="Smart Pricing",
    description="Global Description",
    version="0.0.1"
)

@smart_pricing_api.get("/")
def root():   
    return {"message": "Welcome to NANCY's Smart Pricing"}

@smart_pricing_api.post("/price_calculation")
def calculate_price(payload: ServicesPayload):

    services = payload.services
    provider_agents = []
    providers_max_prices = []
    providers_min_prices = []

    for service in services:
        provider_agents.append(service.provider_id)
        providers_max_prices.append(service.maxprice)
        providers_min_prices.append(service.minprice)


    env_fn = ReverseAuctionEnv 
    env_kwargs = {}
    evaluate(env_fn, num_games=1, render_mode="human", **env_kwargs)
    images_folder = "outputs"
    create_video_from_pngs(images_folder)


    response= {"provider_id":"provider_001", "price":100000, "service_id":"service123"}
    return {"services": response}

@smart_pricing_api.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Invalid payload format", "details": exc.errors()},
    )

@smart_pricing_api.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )



