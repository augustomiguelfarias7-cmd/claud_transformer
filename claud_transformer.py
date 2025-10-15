# cloud_transformer.py
# Cloud Transformer 3.0 - Núcleo da biblioteca
# Copyright (c) 2025 Augusto Miguel de Farias
# Funcionalidades: Modelos Transformers, GitHub, AutoAgent, Agents, OpenAI

import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

# ----------- OpenAI -----------

try:
    import openai
except:
    openai = None

# ----------- Transformers -----------

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except:
    AutoTokenizer = None
    AutoModelForCausalLM = None

# ----------- GitHub -----------

try:
    from github import Github
except:
    Github = None

try:
    from git import Repo
except:
    Repo = None

# ----------- Agents / AutoAgent -----------

import importlib
import shutil
import subprocess

# ================== Transformer Lite ==================
class TransformerLite(nn.Module):
    def __init__(self, vocab_size=30522, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        try:
            x = self.embedding(x)
            x = self.transformer(x)
            logits = self.lm_head(x)
            return logits
        except:
            return None

# ================== GitHub Repositories ==================
class GitHubRepositories:
    def __init__(self, token: str = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.client = Github(self.token) if self.token and Github else None

    def is_configured(self) -> bool:
        return self.client is not None

    def find_repos(self, query: str, max_results: int = 5) -> List[Tuple[str,str]]:
        if not self.client:
            return []
        try:
            results = []
            repos = self.client.search_repositories(query=query, sort="stars", order="desc")
            for r in repos[:max_results]:
                results.append((r.full_name, r.clone_url))
            return results
        except:
            return []

    def clone_repo(self, clone_url: str, destination: str = "./repos"):
        if Repo is None:
            return {"status":"disabled","path":None}
        try:
            os.makedirs(destination, exist_ok=True)
            import pathlib
            repo_name = pathlib.Path(clone_url).stem.replace(".git","")
            dest_path = os.path.join(destination, repo_name)
            if os.path.exists(dest_path):
                return {"status":"exists","path":dest_path}
