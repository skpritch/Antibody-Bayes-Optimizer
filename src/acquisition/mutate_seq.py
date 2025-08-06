# src/acquisition/mutate_seq.py

from __future__ import annotations
import random
import math
from typing import List, Callable
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 AA (no X)

def mutate_sequence(seq: str, n_mutations: int = 1) -> str:
  """
  Randomly mutate `n_mutations` positions in `seq`, avoiding padding 'X'.
  """
  seq_list = list(seq)
  valid_positions = [i for i, aa in enumerate(seq_list) if aa != "X"]
  for _ in range(n_mutations):
      pos = random.choice(valid_positions)
      seq_list[pos] = random.choice([aa for aa in AMINO_ACIDS if aa != seq_list[pos]])
  return ''.join(seq_list)

def hill_climb(
  seed: str,
  fitness_fn: Callable[[str], float],
  local_k2_samples: int = 1000,
  restarts: int = 50,
) -> List[dict]:
  """
  Parallel hill-climb with FULL logging: every neighbourhood eval.
  Returns list of {"sequence","fitness","step","restart"} records.
  """
  records = []
  def _single_hc_log(seed, fn, k2, ridx):
      current = mutate_sequence(seed, 2)
      best = fn(current)
      step = 0
      # log that initial mutated `current`
      records.append(dict(sequence=current, fitness=best,
                          step=0, restart=ridx))
      improved = True
      while improved:
          step += 1
          # build neighbourhood
          neigh = (
            [mutate_sequence(current, 1) for _ in range(len(current)*19)]
            + [mutate_sequence(current, 2) for _ in range(k2)]
          )
          # evaluate & log
          fits = [(fn(s), s) for s in neigh]
          for f, s in fits:
              records.append(dict(sequence=s, fitness=f,
                                  step=step, restart=ridx))
          # pick best
          fmax, smax = max(fits, key=lambda x: x[0])
          if fmax > best:
              current, best = smax, fmax
          else:
              improved = False

  # run restarts in parallel
  Parallel(n_jobs=-1, backend="threading")(
    delayed(_single_hc_log)(seed, fitness_fn, local_k2_samples, r+1)
    for r in range(restarts)
  )
  return records

def genetic_algorithm(
  seed_pop: List[str],
  fitness_fn: Callable[[str], float],
  beta: float = 0.2,
  p_mut: float = 0.05,
  max_gen: int = 100,
  pop_size: int = 200,
) -> List[dict]:
  """
  GA with FULL logging: every generation’s population eval.
  Returns list of {"sequence","fitness","generation"} records.
  """
  records = []
  pop = seed_pop[:pop_size]
  # log initial pop
  for s in pop:
      records.append(dict(sequence=s,
                          fitness=fitness_fn(s),
                          generation=0))
  for g in tqdm(range(1, max_gen+1), desc="GA generations"):
      fits = [fitness_fn(s) for s in pop]
      # log this generation
      for s, f in zip(pop, fits):
          records.append(dict(sequence=s,
                              fitness=f,
                              generation=g))
      # selection + crossover + mutation (same as before)
      probs = np.exp(np.array(fits)/beta)
      probs = (probs / probs.sum()) if probs.sum() else np.ones_like(probs)/len(probs)
      parents = random.choices(pop, weights=probs, k=2*pop_size)
      children = []
      for i in range(0, len(parents), 2):
          p1, p2 = parents[i:i+2]
          cx = random.randint(1, len(p1)-1)
          child = p1[:cx]+p2[cx:]
          child = ''.join(
            random.choice([aa for aa in AMINO_ACIDS if aa!=a]) if random.random()<p_mut else a
            for a in child
          )
          children.append(child)
      pop = children[:pop_size]
  return records

def gibbs_sampling(
  seed: str,
  fitness_fn: Callable[[str], float],
  gamma: float = 20.0,
  iters: int = 30000,
) -> list[dict]:
  """
  Gibbs with FULL logging: every inner iteration’s 20 proposals.
  Returns list of {"sequence","fitness","restart"} records,
  where "restart" is the Gibbs‐iteration index.
  """
  records = []
  current = seed
  L = len(seed)

  for i in tqdm(range(iters), desc="Gibbs iterations"):
      # pick one random site for this iteration
      pos = random.randrange(L)

      # all 20 single‐site mutants at that pos
      cands = [current[:pos] + aa + current[pos+1:] for aa in AMINO_ACIDS]
      fits  = [fitness_fn(s) for s in cands]

      # log each candidate
      for s, f in zip(cands, fits):
          records.append({
              "sequence": s,
              "fitness":  f,
              "restart":  i + 1,
          })

      # now sample a new current from these scores
      raw   = [gamma * f for f in fits]
      mx    = max(raw)
      exps  = [math.exp(r - mx) for r in raw]
      tot   = sum(exps)
      probs = [e / tot for e in exps] if tot else [1/len(exps)] * len(exps)

      aa_new = random.choices(AMINO_ACIDS, weights=probs, k=1)[0]
      if aa_new != current[pos]:
          current = current[:pos] + aa_new + current[pos+1:]

  return records