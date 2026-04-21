🧠 PETZY Model API

Este projeto contém uma API para execução de um modelo de Inteligência Artificial responsável por análise (ex: detecção de dor em animais).

⚙️ Requisitos

Antes de iniciar, você precisa ter instalado:

Python 3
pip (gerenciador de pacotes do Python)
ambiente virtual (venv)
🚀 Configuração do ambiente

Clone o repositório e siga os passos abaixo:

pip install --upgrade pip
pip install -r requirements.txt

python3 -m venv venv
source venv/bin/activate
▶️ Executando a aplicação

Para iniciar o servidor da API:

uvicorn app:app --host 0.0.0.0 --port 8000

A aplicação ficará disponível em:

http://localhost:8000
🤖 Modelo de IA

O modelo .h5 não está versionado no repositório devido a limitações de tamanho do GitHub.

📥 Como obter o modelo

Baixe o modelo manualmente através do link:

https://drive.google.com/file/d/1gR1DZ-IIqfQG77nS_5zlbCvLYbBu9l6g/view?usp=sharing

- Mude o nome para petzy_model.h5 e insira no diretorio raiz do modelo