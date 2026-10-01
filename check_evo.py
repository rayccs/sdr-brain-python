import os
import requests
from dotenv import load_dotenv

def check_evolution():
    load_dotenv('c:/Users/josed/Proyectos/Nodepath/services/SDR_COGNITIVO_B2B/sdr-backend-go/.env')
    evo_url = os.environ.get('EVOLUTION_API_URL')
    evo_key = os.environ.get('EVOLUTION_API_KEY')
    
    headers = {"apikey": evo_key}
    
    try:
        res = requests.get(f"{evo_url}/instance/fetchInstances", headers=headers)
        if res.status_code == 200:
            instances = res.json()
            for inst in instances:
                name = inst.get('name', 'Unknown')
                owner = inst.get('ownerJid', 'Unknown')
                number = inst.get('number', 'Unknown')
                status = inst.get('connectionStatus', 'Unknown')
                print(f"Instance: {name}, Owner: {owner}, Number: {number}, Status: {status}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_evolution()
