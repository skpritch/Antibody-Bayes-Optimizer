# src/developability/biophi_api.py

from __future__ import annotations
import os, json, requests
from time import sleep, time
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.getenv("MERCK_USERNAME")
PASSWORD = os.getenv("MERCK_PASSWORD")
BASE_URL = "https://biophi-dev.merck.com/api"
VERIFY_SSL = False


def run_biophi(
   heavy_chain: str,
   light_chain: str = "",
   name: str = "seq_1",
   pause: int = 30
) -> dict:
   # 1) login
   r = requests.post(
       f"{BASE_URL}/login",
       auth=(USERNAME, PASSWORD),
       verify=VERIFY_SSL
   )
   r.raise_for_status()
   headers = {"Cookie": r.headers["set-cookie"].split(";")[0]}

   # 2) submit
   payload = {"sequences": [{"name": name, "HC": heavy_chain, "LC": light_chain}]}
   sub = requests.post(
       f"{BASE_URL}/reports",
       json=payload,
       headers=headers,
       verify=VERIFY_SSL
   )
   sub.raise_for_status()
   submission = sub.json()

   # 3) poll until done
   summary_url = submission["url_summary"]
   while True:
       sleep(pause)
       status = requests.get(
           summary_url,
           auth=(USERNAME, PASSWORD),
           headers=headers,
           verify=VERIFY_SSL
       ).json()
       if not status.get("in_progress", False):
           break

   # 4) fetch all returned URLs
   result: dict = {}
   for key, url in submission.items():
       if key == "report_id":
           result[key] = url
       else:
           data = requests.get(
               url,
               auth=(USERNAME, PASSWORD),
               headers=headers,
               verify=VERIFY_SSL
           ).json()
           result[key] = data

   return result