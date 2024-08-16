import os
import requests

from dotenv import load_dotenv

load_dotenv()

# PROXYCURL API para profile Scraping
# En verdad, es un proxy para hacer peticiones a un web (Está enfocada en Linkedin)
# y que devuelva la misma en un formato bueno para hacer el scrapping
# https://nubela.co/proxycurl/docs#proxycurl-overview


# URL de pruebas del curso para hacer peticiones a la API sin tener que gastar créditos
# página gist.github para hacer peticiones sin gastar créditos de la api de LINKEDIN
# https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json
# Puedo utilizar mi dirección de linkedin también
def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """scrape infomration from linkedIn profiles,
    Manually scrpae the information from the LinkedIn profile"""

    # Si queremos tirar del perfil mockeado de linkedin, pues dejamos mock False
    # caso contrario mock True
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
        response = requests.get(linkedin_profile_url, timeout=10)
    else:
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        params = {"url": linkedin_profile_url}
        headers = {"Authorization": "Bearer " + os.getenv("PROXYCURL_API_KEY")}
        response = requests.get(
            api_endpoint, params=params, headers=headers, timeout=10
        )

    data = response.json()
    # Nos quedamos con cierta información de la respuesta, no con toda.
    # Hacemos compresión de listas ( en este caso de diccionarios, para cada valor de data, y ponemos key value)
    data = {
        k: v for k,v in data.items() if v not in ([],"","", None) and k not in ["people_also_viewed","certifications"]
    }
    
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")
    
    return data



# --------------------------------------------------
if __name__ == "__main__":
    print(
        scrape_linkedin_profile(
            linkedin_profile_url="www.linkedin.com/in/domingo-lópez-pacheco-957831139",
            mock=True
        )
    )
