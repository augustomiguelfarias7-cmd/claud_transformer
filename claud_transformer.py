# cloudtransformer.py
# Núcleo da biblioteca Cloud Transformer 🚀
# ~ 200 linhas incluindo os 15 modelos

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


class CloudTransformer:
    """
    Núcleo que conecta os modelos disponíveis.
    Uso:
        from cloud_transformer import CloudTransformer
        ct = CloudTransformer()
        model = ct.load("gpt2")
        print(model.generate("Olá mundo"))
    """

    def __init__(self):
        self.supported = {
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
        }

    def available_models(self):
        """Lista os modelos disponíveis"""
        return list(self.supported.keys())

    def load(self, name, **kwargs):
        """Carrega um modelo pelo nome"""
        if name not in self.supported:
            raise ValueError(f"Modelo '{name}' não encontrado! Use: {self.available_models()}")
        return self.supported[name](**kwargs)


# ============= MODELS ============= #
# Exemplos simplificados dentro da pasta /models/
# Aqui vou mostrar 5 de exemplo, os outros seguem o mesmo padrão.

# models/gpt2.py
from transformers import pipeline

class GPT2Model:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt2")

    def generate(self, prompt, max_length=50):
        return self.generator(prompt, max_length=max_length)[0]["generated_text"]


# models/whisper.py
from transformers import pipeline

class WhisperModel:
    def __init__(self):
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    def transcribe(self, audio_path):
        return self.transcriber(audio_path)["text"]


# models/stable_diffusion.py
from diffusers import StableDiffusionPipeline
import torch

class StableDiffusionModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.pipe = self.pipe.to(device)

    def generate(self, prompt, filename="output.png"):
        image = self.pipe(prompt).images[0]
        image.save(filename)
        return filename


# models/controlnet.py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel as CN
import torch

class ControlNetModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        controlnet = CN.from_pretrained("lllyasviel/sd-controlnet-canny")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet
        ).to(device)

    def generate(self, prompt, image_condition, filename="controlnet.png"):
        result = self.pipe(prompt, image=image_condition).images[0]
        result.save(filename)
        return filename


# models/clip.py
from transformers import CLIPProcessor, CLIPModel as HFCLIP
from PIL import Image

class CLIPModel:
    def __init__(self):
        self.model = HFCLIP.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def similarity(self, image_path, texts):
        image = Image.open(image_path)
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs.tolist()
