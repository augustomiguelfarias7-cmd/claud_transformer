# cloud_transformer.py
# Núcleo da biblioteca Cloud Transformer 🚀
# Versão 2.0 - Integrando modelos antigos, OpenAI e suporte a modelos do usuário
# ~250 linhas incluindo todos os modelos e funcionalidades

import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPProcessor, CLIPModel as HFCLIP
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel as CN
from PIL import Image
from datasets import load_dataset
import openai

# ================== MODELOS ANTIGOS ================== #

from .models.gpt2 import GPT2Model
from .models.gptneo import GPTNeoModel
from .models.gptj import GPTJModel
from .models.t5 import T5Model
from .models.bart import BARTModel
from .models.bert import BERTModel
from .models.roberta import RoBERTaModel
from .models.distilbert import DistilBERTModel
from .models.whisper import WhisperModel
from .models.wav2vec2 import Wav2Vec2Model
from .models.clip import CLIPModel
from .models.dalle_mini import DALLEMiniModel
from .models.stable_diffusion import StableDiffusionModel
from .models.controlnet import ControlNetModel
from .models.bloom import BLOOMModel

# ================== CLOUD TRANSFORMER ================== #

class CloudTransformer:
    """
    Núcleo que conecta os modelos disponíveis.
    Uso:
        from cloud_transformer import CloudTransformer
        ct = CloudTransformer()
        model = ct.load("gpt2")
        print(model.generate("Olá mundo"))
    """

    def __init__(self, openai_api_key=None):
        # Chave API OpenAI
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key

        self.supported = {
            # Modelos antigos
            "gpt2": GPT2Model,
            "gpt-neo": GPTNeoModel,
            "gpt-j": GPTJModel,
            "t5": T5Model,
            "bart": BARTModel,
            "bert": BERTModel,
            "roberta": RoBERTaModel,
            "distilbert": DistilBERTModel,
            "whisper": WhisperModel,
            "wav2vec2": Wav2Vec2Model,
            "clip": CLIPModel,
            "dalle-mini": DALLEMiniModel,
            "stable-diffusion": StableDiffusionModel,
            "controlnet": ControlNetModel,
            "bloom": BLOOMModel,
            # Modelos OpenAI
            "gpt-3.5-turbo": self._GPTOpenAI,
            "gpt-4": self._GPTOpenAI,
            "gpt-4-turbo": self._GPTOpenAI,
            "dalle-2": self._DALLEOpenAI,
            "dalle-3": self._DALLEOpenAI,
        }

    def available_models(self):
        """Lista os modelos disponíveis"""
        return list(self.supported.keys())

    def load(self, name, **kwargs):
        """Carrega um modelo pelo nome"""
        if name not in self.supported:
            raise ValueError(f"Modelo '{name}' não encontrado! Use: {self.available_models()}")
        model_class = self.supported[name]
        if name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
            return model_class(name, self.openai_api_key, **kwargs)
        elif name in ["dalle-2", "dalle-3"]:
            return model_class(name, self.openai_api_key, **kwargs)
        else:
            return model_class(**kwargs)

    # ================== MÓDULOS OPENAI ================== #
    class _GPTOpenAI:
        def __init__(self, model_name, api_key, **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            openai.api_key = api_key

        def generate(self, prompt, max_tokens=150, temperature=0.7):
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

    class _DALLEOpenAI:
        def __init__(self, model_name, api_key, **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            openai.api_key = api_key

        def generate(self, prompt, filename="output.png"):
            response = openai.Image.create(
                prompt=prompt,
                model=self.model_name,
                size="1024x1024"
            )
            image_data = response.data[0].b64_json
            import base64
            img_bytes = base64.b64decode(image_data)
            with open(filename, "wb") as f:
                f.write(img_bytes)
            return filename

    # ================== MODELOS DO USUÁRIO ================== #
    def load_user_model(self, model_name, weights_path, tokenizer_path=None, **kwargs):
        """Carrega um modelo do usuário fornecendo caminho dos pesos"""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or weights_path)
        model = AutoModelForCausalLM.from_pretrained(weights_path)
        return {"name": model_name, "model": model, "tokenizer": tokenizer}

    # ================== DATASETS ================== #
    def load_dataset(self, dataset_name, split="train"):
        """Carrega datasets da Hugging Face"""
        ds = load_dataset(dataset_name, split=split)
        return ds
