from setuptools import setup, find_packages

# --- Lógica de Leitura Segura do README.md ---
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Consulte o repositório GitHub para a descrição completa."
# --- Fim da Lógica de Leitura Segura ---

setup(
    name="cloud_transformer",
    version="3.0.0",
    author="Augusto Miguel de Farias",
    author_email="augustomiguelfarias@gmail.com",
    description="Biblioteca Python para trabalhar com modelos Transformers, geração de texto, imagem, áudio, vídeo e integração OpenAI/GitHub/Agentes.",
    
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    url="https://github.com/augustomiguelfarias7-cmd/claud_transformer.git",

    py_modules=["cloud_transformer"],
    packages=find_packages(),

    # Dependências obrigatórias
    install_requires=[
        "torch>=2.0",
        "transformers>=4.0",
        # ALTERAÇÃO: Mudança da restrição de versão para a mais antiga encontrada (0.0.1)
        "diffusers>=0.0.1", 
        "Pillow>=4.0",
        "requests>=2.30",
        "openai>=1.0",
        "datasets>=2.0",
        "PyGithub>=1.0",
        "GitPython>=3.1"
    ],

    # Extras opcionais
    extras_require={
        # ALTERAÇÃO: Mudança da restrição de versão para a mais antiga encontrada (0.0.1)
        "vision": ["diffusers>=0.0.1", "Pillow>=9.0"], # geração de imagens
        "text": ["transformers>=4.0"],
        "audio": ["torchaudio>=2.1"],
        "video": ["opencv-python>=4.7"],
        "agents": ["auto-gpt>=0.1"]
    },

    python_requires='>=3.10',

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
