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
    service_name = services[0].service_id
    for service in services:
        possible_agents.append(service.provider_id)
        initial_prices.append(service.maxprice)
        providers_min_prices.append(service.minprice)

    avg_min = sum(providers_min_prices) / len(providers_min_prices)
    num_bidders = len(possible_agents)
    max_rounds = 10

    config_file_path = os.path.join('spm', 'config.yml')
    winner_agents_file_path = os.path.join('spm', 'outputs', 'winner_agents.json')

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

   

    subprocess.run(['python', 'main.py', '--mode', 'evaluate'], cwd='spm')

    results_path = os.path.join('spm', 'auction_results.json')


    with open(results_path, 'r') as json_file:
        auction_result = json.load(json_file)
    
    with open(winner_agents_file_path, 'r') as winner_json_file:
        winners_data = json.load(winner_json_file)
        print(winners_data)
    
    
    ######### SAFETY MECHANISM TO ENSURE THE FINAL PRICE DOES NOT FALL UNDER A MINIMUM PRICE #########
    if auction_result['price'] < providers_min_prices[possible_agents.index(auction_result['winner'])]:
        #print(f"Providers min price is {providers_min_prices[possible_agents.index(auction_result['winner'])]} and auction result is {auction_result['price']}")
        for winner in reversed(winners_data):
            if winner['bid'] >= providers_min_prices[possible_agents.index(winner['agent'])]:
                print(f"Winner is {winner['agent']} with price {winner['bid']} and min price {providers_min_prices[possible_agents.index(winner['agent'])]}")
                auction_result['winner'] = winner['agent']
                auction_result['price'] = winner['bid']
                break
    ######### END OF SAFETY MECHANISM #########

    response= {"provider_id":auction_result['winner'], "price":auction_result['price'], "service_id":service_name}
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



