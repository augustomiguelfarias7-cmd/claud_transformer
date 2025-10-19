from setuptools import setup, find_packages

setup(
    name="cloud_transformer",  # Nome do pacote correto
    version="3.0.0",           # Versão atualizada
    author="Augusto Miguel de Farias",
    author_email="augustomiguelfarias@gmail.com",
    description="Biblioteca Python para trabalhar com modelos Transformers, geração de texto, imagem, áudio, vídeo e integração OpenAI/GitHub/Agentes.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/augustomiguelfarias7-cmd/claud_transformer.git",

    # Arquivo principal
    py_modules=["cloud_transformer"],  
    packages=find_packages(),

    # Dependências obrigatórias
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.0.",
        "diffusers>=4.0",
        "Pillow>=4.0",
        "requests>=2.30",
        "openai>=1.0",
        "datasets>=2.0",
        "PyGithub>=1.0",
        "GitPython>=3.1"
    ],

    # Extras opcionais
    extras_require={
        "vision": ["diffusers>=4.0", "Pillow>=9.0"],  # geração de imagens
        "text": ["transformers>=4.0"],                # modelos de texto
        "audio": ["torchaudio>=2.1"],                # suporte a áudio
        "video": ["opencv-python>=4.7"],             # suporte a vídeo futuro
        "agents": ["auto-gpt>=0.1"]                  # integração com AutoAgent / AutoGPT
    },

    python_requires='>=3.10',

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

