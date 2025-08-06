# src/developability/dev_score.py

from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from developability.biophi_api import run_biophi

# Default cache directory
DEV_DIR = Path("../data/biophi")
ALPHA = 1.0
BETA  = 0.1

def get_or_create_dev_json(
  seq_id: int,
  heavy_chain: str,
  light_chain: str = "",
  name: str | None = None,
  dev_dir: Path = DEV_DIR
) -> Dict[str, Any]:
  """
  Look for any file dev_dir/{seq_id}_*.json.
  If found, load & return it.
  Otherwise, scan seq_id-40…seq_id+40 for an identical sequence
  and reuse its JSON if we find one. Failing that, call run_biophi().
  """
  dev_dir = Path(dev_dir)
  dev_dir.mkdir(parents=True, exist_ok=True)


  # 1) exact‐match cache by seq_id
  pattern = f"{seq_id}_*.json"
  matches = list(dev_dir.glob(pattern))
  if matches:
      dev_json = json.loads(matches[0].read_text())
      top_key = next(k for k in dev_json if k not in ("report_id", "url_summary"))
      if top_key == heavy_chain:
           return dev_json
      return json.loads(matches[0].read_text())


  # 2) no exact match → scan nearby IDs for identical heavy_chain
  start = max(0, seq_id - 40)
  end   = seq_id + 40
  for other_id in range(start, end + 1):
      if other_id == seq_id:
          continue
      for path in dev_dir.glob(f"{other_id}_*.json"):
          dev_json = json.loads(path.read_text())
          # top_key is the heavy_chain string under which run_biophi nested its results
          top_key = next(k for k in dev_json if k not in ("report_id", "url_summary"))
          if top_key == heavy_chain:
              # identical sequence found—reuse JSON
              return dev_json


  # 3) still not found → call API and cache
  merged = run_biophi(heavy_chain, light_chain, name=name)
  report_id = merged["report_id"]
  fname = f"{seq_id}_{report_id}.json"
  with open(dev_dir / fname, "w") as f:
      json.dump(merged, f, indent=2)
  return merged


def parse_dev_json(dev_json: Dict[str, Any]) -> Dict[str, Any]:
   top_key = next(k for k in dev_json if k not in ("report_id", "url_summary"))
   overview = dev_json[top_key].get("overview", {})
   cdrs_pi  = overview.get("cdrs_pi", None)

   summary = dev_json["url_summary"]["results"][0]
   comp    = summary["composition"]
   flags   = summary["flags"]
   human   = summary["humanness"]

   def flatten(d):
       out = []
       for items in d.values():
           out.extend(item["type"] for item in (items or []))
       return out

   return {
       "framework_liabilities":    flatten(comp["framework_liabilities"]),
       "primary_cdr_liabilities":  flatten(comp["primary_cdr_liabilities_(kabat)"]),
       "secondary_cdr_liabilities":flatten(comp["secondary_cdr_liabilities"]),
       "human_peptide_content":    human["human_peptide_content"],
       "human_germline_content":   human["human_germline_content"],
       "cdrs_pi":                  cdrs_pi,
       "poly_specificity_risk":    flags["%_poly-specificity_risk"],
       "vhh_positive_patch_area":  flags["vhh_positive_patch_area_(ph_7.4)"],
   }


def compute_dev_score(metrics: Dict[str, Any]) -> float:
   n1 = len(metrics["primary_cdr_liabilities"])
   n2 = len(metrics["secondary_cdr_liabilities"])
   penalty = ALPHA * n1 + BETA * n2
   liab_score = float(np.exp(-penalty**2))

   hpc = metrics["human_peptide_content"]
   if hpc <= 0.65:
       hpc_score = 0.0
   elif hpc >= 0.80:
       hpc_score = 1.0
   else:
       hpc_score = (hpc - 0.65) / 0.15

   pi = metrics["cdrs_pi"]
   if pi is None:
       pi_score = 0.0
   elif 7.5 <= pi <= 9.0:
       pi_score = 1.0
   elif pi < 7.5:
       pi_score = max(0.0, 1 - (7.5 - pi) / 2.5)
   else:
       pi_score = max(0.0, 1 - (pi - 9.0) / 3.0)

   r = metrics["poly_specificity_risk"]
   v = metrics["vhh_positive_patch_area"]
   nr = max(0.0, 1 - r / 100)
   nv = max(0.0, 1 - v / 3000)
   psr_vhh_score = (nr + nv) / 2

   return 2 * (0.45 * liab_score + 0.275 * hpc_score + 0.175 * pi_score + 0.1 * psr_vhh_score)


def score_sequences(
   seqs: List[str],
   seq_ids: List[int],
   light_chain: str = "",
   names: List[str] | None = None,
   dev_dir: Path = DEV_DIR
) -> List[Optional[float]]:
   """
   Batch wrapper: for each (seq, seq_id), fetch or call API, parse, score.
   """

   if names is None:
       names = [None] * len(seqs)

   scores: List[Optional[float]] = []
   for hc, sid, nm in zip(seqs, seq_ids, names):
       try:
           dev_json = get_or_create_dev_json(sid, hc, light_chain, nm, dev_dir)
           metrics  = parse_dev_json(dev_json)
           scores.append(compute_dev_score(metrics))
       except Exception as e:
           print(f"Warning: developability scoring failed for seq_id={sid}: {e}")
           scores.append(float("nan"))
   return scores