from fastapi import FastAPI, Depends, Response, status, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from endpoint_classes import ServicesPayload
from pydantic import ValidationError
import os


import yaml
import subprocess
import json




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
    possible_agents = []
    initial_prices = []
    providers_min_prices = []

    for service in services:
        possible_agents.append(service.provider_id)
        initial_prices.append(service.maxprice)
        providers_min_prices.append(service.minprice)

    avg_min = sum(providers_min_prices) / len(providers_min_prices)
    num_bidders = len(possible_agents)
    max_rounds = 20

    config_file_path = os.path.join('spm', 'config.yml')


    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    # Update the YAML file with the calculated values
    config_data['environment']['num_bidders'] = num_bidders
    config_data['environment']['max_rounds'] = max_rounds
    config_data['environment']['avg_min'] = avg_min
    config_data['environment']['initial_prices'] = initial_prices
    config_data['environment']['possible_agents'] = possible_agents

    # Write the updated data back to the YAML file
    with open(config_file_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)

    print("ok")

    subprocess.run(['python', 'main.py', '--mode', 'evaluate'], cwd='spm')

    results_path = os.path.join('spm', 'auction_results.json')


    with open(results_path, 'r') as json_file:
        data = json.load(json_file)
    

    response= {"provider_id":data['winner'], "price":data['price'], "service_id":"service123"}
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



