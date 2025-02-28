# RAG API

Uma API simples implementando Retrieval-Augmented Generation (RAG) com FastAPI, Sentence Transformers e suporte para dois providers de LLM: OpenAI e Ollama rodando local.

## Visão Geral

Este projeto fornece uma API que:
1. Armazena documentos de texto em memória
2. Converte consultas e documentos em embeddings usando Sentence Transformers
3. Encontra o documento mais relevante para uma consulta usando similaridade de cosseno
4. Gera uma resposta baseada no documento encontrado usando OpenAI ou Ollama local

## Requisitos

- Python 3
- 4GB+ VRAM para rodar o modelo llama3.2 no Ollama localmente
- 16GB+ RAM para operações com sentence-transformers

## Instalação

1. Clone o repositório:
   ```bash
   git clone
   cd rag-api
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure as variáveis de ambiente:
   ```bash
   export OPENAI_API_KEY="chave"
   ```

## Configurando o Ollama (para endpoint /queryLocal)

1. Instale o Ollama seguindo as instruções em [ollama.ai](https://ollama.ai/)
2. Baixe o modelo llama3.2:
   ```bash
   ollama pull llama3.2
   ```
3. Certifique-se de que o servidor Ollama esteja rodando antes de usar o endpoint `/queryLocal`

## Uso

1. Inicie o servidor:
   ```bash
    uvicorn fastrag:app --host 0.0.0.0 --port 8000 --reload
   ```

2. A API estará disponível em `http://localhost:8000`

### Endpoints

- **POST /query**: Consulta usando OpenAI
  ```bash
  curl -X POST "http://localhost:8000/query" \
       -H "Content-Type: application/json" \
       -d '{"query": "Quantos corações um polvo tem?"}'
  ```

- **POST /queryLocal**: Consulta usando Ollama localmente
  ```bash
  curl -X POST "http://localhost:8000/queryLocal" \
       -H "Content-Type: application/json" \
       -d '{"query": "Quantos corações um polvo tem?"}'
  ```

## Documentos de Exemplo

A API vem pré-carregada com 5 documentos de exemplo contendo fatos curiosos:
1. Sobre corações de polvos
2. Sobre a Grande Muralha da China
3. Sobre mel
4. Sobre o dia em Vênus
5. Sobre um nome de cachorro

## Arquitetura

O aplicativo utiliza um padrão RAG (Retrieval-Augmented Generation) simples:
1. Os documentos são codificados em embeddings usando Sentence Transformers
2. As consultas dos usuários são convertidas em embeddings da mesma forma
3. O sistema encontra o documento mais similar à consulta
4. O LLM (OpenAI ou Ollama) gera uma resposta baseada no documento recuperado

## Limitações

- Armazenamento de documentos em memória (não persistente)
- Sem otimização para grandes conjuntos de documentos
- Interface básica sem paginação ou filtragem avançada