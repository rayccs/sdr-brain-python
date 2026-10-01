import os
import requests
from dotenv import load_dotenv

def delete_instance():
    load_dotenv('c:/Users/josed/Proyectos/Nodepath/services/SDR_COGNITIVO_B2B/sdr-backend-go/.env')
    evo_url = os.environ.get('EVOLUTION_API_URL')
    evo_key = os.environ.get('EVOLUTION_API_KEY')
    
    headers = {"apikey": evo_key}
    
    try:
        # Delete the duplicate instance
        res = requests.delete(f"{evo_url}/instance/delete/sdr-raymonf-epidataconsulting-com", headers=headers)
        print(f"Deleted sdr-raymonf-epidataconsulting-com: {res.status_code} {res.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    delete_instance()
