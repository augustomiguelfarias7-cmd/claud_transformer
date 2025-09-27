

# Cloud Transformer 🚀 v2.0

Cloud Transformer é uma biblioteca Python para trabalhar com modelos Transformers de texto, áudio, imagem, vídeo e integração OpenAI. Agora com suporte para treinar modelos, criar tokenizers, usar datasets e adicionar modelos próprios.


---

Funcionalidades Principais

Carregar modelos de linguagem (GPT-2, GPT-Neo, GPT-J, T5, BART, BERT, RoBERTa, DistilBERT, BLOOM).

Carregar modelos multimodais (CLIP, DALL·E Mini, DALL·E2/3, Stable Diffusion, ControlNet).

Suporte a modelos de áudio (Whisper, Wav2Vec2).

Integração OpenAI: GPT-3.5, GPT-4, GPT-4 Turbo, DALL·E2/3.

Criar tokenizers personalizados.

Treinar e fine-tunar modelos.

Adicionar modelos próprios fornecendo apenas o caminho dos pesos.

Suporte a datasets (via datasets da Hugging Face).

Compatível com Python >= 3.10 e CUDA se disponível.



---

Instalação

pip install git+https://github.com/augustomiguelfarias7-cmd/claud_transformer.git


---

Exemplos de Uso

1. Carregar modelos de texto

from cloud_transformer import CloudTransformer

ct = CloudTransformer()
gpt2 = ct.load("gpt2")
texto = gpt2.generate("Era uma vez um robô que")
print(texto)


---

2. Usar modelos da OpenAI

# Usando GPT-4 e GPT-3.5 Turbo
openai_model = ct.load("openai", api_key="SUA_CHAVE_OPENAI", model="gpt-4")
resposta = openai_model.generate("Explique a Revolução Industrial")
print(resposta)

# Usando DALL·E 3
dalle3 = ct.load("openai", api_key="SUA_CHAVE_OPENAI", model="dall-e-3")
dalle3.generate("Um dragão futurista pintando uma obra de arte")


---

3. Criar tokenizer personalizado

from cloud_transformer import TokenizerManager

tokenizer = TokenizerManager.create_tokenizer("meu_tokenizer")
tokenizer.train_from_dataset("caminho/do/dataset.txt")


---

4. Treinar ou fine-tunar modelo

from cloud_transformer import Trainer

trainer = Trainer(model_name="gpt2")
trainer.train(dataset_path="caminho/do/dataset")
trainer.save_model("meu_modelo_finetunado")


---

5. Adicionar e usar modelo próprio

meu_modelo = ct.load_custom("meu_modelo", path="caminho/pesos/modelo.pt")
texto = meu_modelo.generate("Olá mundo!")
print(texto)
