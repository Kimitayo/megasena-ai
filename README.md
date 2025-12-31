# 🧠 Mega-Sena AI: Sistema Híbrido de Previsão (Ensemble Deep Learning)

Este projeto é um sistema avançado de inteligência artificial desenvolvido para análise e previsão de resultados da loteria Mega-Sena. 

Diferente de geradores aleatórios simples, este sistema utiliza uma abordagem de **Nível 5 (Híbrida)**, combinando Redes Neurais Recorrentes (LSTM e GRU) com Filtragem Estatística Rigorosa e Mineração de Dados.

## 🚀 Funcionalidades Principais

O sistema opera em três camadas de inteligência:

1.  **Ensemble Learning (O Comitê de IAs):**
    * Utiliza dois modelos distintos de Deep Learning: **Bidirectional LSTM** (Long Short-Term Memory) e **Bidirectional GRU** (Gated Recurrent Unit).
    * As previsões são geradas por consenso entre as duas redes para reduzir o viés.

2.  **Feature Engineering (Engenharia de Atributos):**
    * A IA não analisa apenas os números brutos. Ela é treinada com 10 dimensões extras de dados, incluindo:
        * Distribuição de Pares/Ímpares.
        * Soma total das dezenas.
        * Números Primos e Fibonacci.
        * Distribuição Espacial (Quadrantes do bilhete).
        * Amplitude (Distância entre o menor e maior número).

3.  **Filtro Híbrido de Elite (Otimização Combinatória):**
    * A IA seleciona um "pool" das 18 dezenas mais prováveis.
    * Um algoritmo matemático gera todas as combinações possíveis entre elas e aplica filtros estatísticos (ex: descarta jogos com soma absurda ou sem primos).
    * O resultado é um ranking dos jogos matematicamente mais viáveis.

## 📂 Estrutura do Projeto

* `main.py`: O "cérebro". Responsável por processar o CSV, calcular as estatísticas avançadas e treinar os dois modelos (LSTM e GRU). Salva os arquivos `.keras`.
* `app.py`: O "oráculo". Carrega os modelos treinados, faz a previsão do próximo jogo, aplica o filtro combinatório e exibe o ranking de probabilidades (Zona Quente, Morna e Fria).
* `analise_padroes.py`: Ferramenta de **Data Mining**. Analisa o histórico completo para encontrar pares frequentes, regras de associação ("Se sai X, sai Y") e números atrasados.
* `mega_sena-1.csv`: Base de dados histórica (deve ser atualizada periodicamente).

## 🛠️ Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/mega-sena-ai.git](https://github.com/SEU-USUARIO/mega-sena-ai.git)
    cd mega-sena-ai
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No Linux/Mac:
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install numpy pandas tensorflow scikit-learn
    ```

## 🧠 Como Usar

### Passo 1: Treinar a Inteligência
Sempre que você atualizar o arquivo `mega_sena-1.csv` com novos jogos, execute o treinamento para atualizar os cérebros neurais:

```bash
python main.py