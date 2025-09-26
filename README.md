Cloud Transformer 🚀

Cloud Transformer é uma biblioteca Python para trabalhar com modelos Transformers, incluindo texto, multimodal, áudio e imagens. Ela oferece uma interface unificada para carregar e usar modelos populares, incluindo BLOOM, GPT2, Whisper, Stable Diffusion, ControlNet, CLIP e outros.

Funcionalidades principais

Carregar modelos de linguagem, multimodal e visão facilmente.

Suporte a 15 modelos principais, todos acessíveis diretamente pelo Cloud Transformer.

Geração de texto, transcrição de áudio, criação de imagens e análise multimodal.

Suporte a GPU ou CPU, com verificação automática de dispositivos.

Compatível com Python 3.10+.



---

Modelos disponíveis

Modelo	Tipo	Biblioteca usada

GPT2	Texto	transformers
GPT-Neo	Texto	transformers
GPT-J	Texto	transformers
T5	Texto	transformers
BART	Texto	transformers
BERT	Texto	transformers
RoBERTa	Texto	transformers
DistilBERT	Texto	transformers
Whisper	Áudio	transformers
Wav2Vec2	Áudio	transformers
CLIP	Multimodal	transformers, Pillow
DALL·E Mini	Imagem	diffusers, torch
Stable Diffusion	Imagem	diffusers, torch
ControlNet	Imagem	diffusers, torch
BLOOM	Texto	transformers



---

Dependências da biblioteca

torch>=2.0 – para todos os modelos baseados em PyTorch.

transformers>=4.0 – modelos de linguagem e multimodal.

diffusers – modelos de geração de imagens (Stable Diffusion, ControlNet, DALL·E Mini).

Pillow – manipulação de imagens para CLIP e pipelines de imagem.

requests – comunicação com APIs e downloads de modelos.



---

Instalação

pip install git+https://github.com/augustomiguelfarias7-cmd/claud_transformer.git


---

Exemplo de uso

from cloud_transformer import CloudTransformer

# Inicializa a biblioteca
ct = CloudTransformer()

# Lista todos os modelos disponíveis
print("Modelos disponíveis:", ct.available_models())

# Carrega o modelo BLOOM
bloom_model = ct.load("bloom")

# Gera texto com o BLOOM
prompt = "Era uma vez um dinossauro que"
texto_gerado = bloom_model.generate(prompt, max_length=100)
print("Texto gerado pelo BLOOM:", texto_gerado)

# Carrega modelo de áudio (Whisper) e transcreve
whisper_model = ct.load("whisper")
texto_audio = whisper_model.transcribe("audio_exemplo.wav")
print("Transcrição do áudio:", texto_audio)

# Carrega modelo de imagem (Stable Diffusion)
sd_model = ct.load("stable-diffusion")
sd_model.generate("Um robô pintando uma obra de arte futurista")
