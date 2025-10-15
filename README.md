Cloud Transformer 3.0 🚀

Cloud Transformer é uma biblioteca Python para trabalhar com modelos Transformers, integração com OpenAI GPT, busca e clonagem de repositórios GitHub, criação de agentes e uso de Transformer Lite. Tudo funcional e preparado para estudo, testes e projetos.


---

Instalação:

Para instalar diretamente do GitHub, rode:

pip install git+https://github.com/augustomiguelfarias7-cmd/claud_transformer.git

Dependências principais:
Python >= 3.10, torch >= 2.0, transformers >= 4.0, diffusers >= 1.0, Pillow >= 9.0, requests >= 2.30, openai >= 1.0, datasets >= 2.0, PyGithub >= 1.0, GitPython >= 3.1


---

Funcionalidades principais:

Transformer Lite — criar modelos Transformer leves próprios.

Modelos do usuário — carregar modelos Transformers customizados.

GitHub Repositories — buscar e clonar repositórios via token.

Agentes — criar agentes simples para automatizar respostas.

AutoAgent / AutoGPT — integração com agentes inteligentes.

OpenAI GPT-3/3.5/4 — geração de texto com modelos da OpenAI.



---

Exemplo de uso:

Inicialize a biblioteca com suas chaves da OpenAI e GitHub:

from cloud_transformer import CloudTransformer

ct = CloudTransformer(
    openai_api_key="YOUR_OPENAI_KEY",
    github_token="YOUR_GITHUB_TOKEN"
)

Usando GPT-2:

gpt2_model = ct.load_user_model("gpt2", weights_path="gpt2")
tokenizer = gpt2_model["tokenizer"]
model = gpt2_model["model"]

input_text = "Olá mundo! Este é um teste com GPT-2."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print("GPT-2:", tokenizer.decode(outputs[0], skip_special_tokens=True))

Usando GPT-3.5 Turbo:

gpt3_model = ct.OpenAIModel("gpt-3.5-turbo", ct.openai_api_key)
response = gpt3_model.generate("Escreva um pequeno poema sobre robôs e inteligência artificial.")
print("GPT-3.5 Turbo:", response)


---

Licença:

Cloud Transformer 3.0 é distribuída com a CTCL (Cloud Transformer Community License).
Permite: uso comercial, estudo, modificação e redistribuição com a mesma licença.
Proibido uso indevido ou sexual sem autorização do autor.

