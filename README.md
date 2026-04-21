# 🧠 PETZY Model API

Esta API fornece uma interface para a execução de modelos de Inteligência Artificial voltados para a análise de bem-estar animal, com foco na detecção de dor e estados fisiológicos através de processamento de imagem.

## ⚙️ Requisitos

Antes de iniciar, certifique-se de ter instalado em sua máquina:

* **Python 3.8+**
* **pip** (Gerenciador de pacotes do Python)
* **venv** (Módulo para ambientes virtuais)

## 🚀 Configuração do Ambiente

Siga os passos abaixo para preparar o ambiente de desenvolvimento:

1.  **Criar o ambiente virtual:**
    ```bash
    python3 -m venv venv
    ```

2.  **Ativar o ambiente virtual:**
    * No Linux/macOS (Pop!_OS):
        ```bash
        source venv/bin/activate
        ```
    * No Windows:
        ```bash
        .\venv\Scripts\activate
        ```

3.  **Atualizar o pip e instalar dependências:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## ▶️ Executando a Aplicação

Para iniciar o servidor da API utilizando o Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
A aplicação estará disponível em: http://localhost:8000

Dica: Você pode acessar a documentação automática da API (Swagger UI) em http://localhost:8000/docs.
```

## 🤖 Modelo de IA
O arquivo do modelo (.h5) não está versionado neste repositório devido às limitações de tamanho do GitHub e para manter o repositório leve.

📥 Como obter o modelo
Baixe o modelo manualmente através do link abaixo:

[Download petzy_model.h5 (Google Drive)](https://drive.google.com/file/d/1gR1DZ-IIqfQG77nS_5zlbCvLYbBu9l6g/view?usp=drive_link)

Após o download, renomeie o arquivo para petzy_model.h5 (caso necessário).

Insira o arquivo no diretório raiz deste projeto.
