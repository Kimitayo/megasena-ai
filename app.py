import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
import os
from itertools import combinations

# CONFIGURAÇÕES
ARQUIVO_DADOS = 'mega_sena-1.csv'
ARQUIVO_LSTM = 'cerebro_lstm.keras'
ARQUIVO_GRU = 'cerebro_gru.keras'
WINDOW_SIZE = 20
NUM_NUMEROS = 60
NUM_FEATURES_EXTRAS = 10
INPUT_DIM = NUM_NUMEROS + NUM_FEATURES_EXTRAS

# Parâmetros do Filtro 
MIN_SOMA = 140
MAX_SOMA = 240
MIN_PARES = 2
MAX_PARES = 4
MIN_PRIMOS = 1
MIN_FIBO = 0 
MIN_AMPLITUDE = 20 

PRIMOS = set([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55])

def carregar_ultimos_jogos():
    dataset_final = []
    if not os.path.exists(ARQUIVO_DADOS): return []
    try:
        with open(ARQUIVO_DADOS, 'r', encoding='utf-8', errors='ignore') as f:
            linhas = f.readlines()
            for linha in linhas:
                linha = linha.strip()
                if not linha: continue
                partes = linha.split(',') if ',' in linha else linha.split(';')
                nums = []
                for p in partes:
                    try:
                        n = int(p)
                        if 1 <= n <= 60: nums.append(n)
                    except: continue
                if len(nums) >= 6:
                    dataset_final.append(sorted(nums[-6:]))
        if len(dataset_final) >= WINDOW_SIZE:
            return dataset_final[-WINDOW_SIZE:]
        else: return []
    except: return []

def calcular_features_extras(jogos):
    features = []
    max_soma = 345
    for i in range(len(jogos)):
        jogo = jogos[i]
        qtd_pares = sum(1 for n in jogo if n % 2 == 0)
        soma = sum(jogo)
        qtd_primos = sum(1 for n in jogo if n in PRIMOS)
        qtd_fibo = sum(1 for n in jogo if n in FIBONACCI)
        repetidos = sum(1 for n in jogo if n in set(jogos[i-1])) if i > 0 else 0
        q1 = q2 = q3 = q4 = 0
        for n in jogo:
            if n <= 30:
                if n % 10 != 0 and (n % 10) <= 5: q1 += 1
                else: q2 += 1
            else:
                if n % 10 != 0 and (n % 10) <= 5: q3 += 1
                else: q4 += 1
        amplitude = max(jogo) - min(jogo)
        features.append([
            qtd_pares/6.0, soma/max_soma, qtd_primos/6.0, qtd_fibo/6.0, repetidos/6.0,
            q1/6.0, q2/6.0, q3/6.0, q4/6.0, amplitude/59.0
        ])
    return np.array(features)

def validar_jogo(jogo):
    soma = sum(jogo)
    pares = sum(1 for n in jogo if n % 2 == 0)
    primos = sum(1 for n in jogo if n in PRIMOS)
    fibo = sum(1 for n in jogo if n in FIBONACCI)
    amplitude = max(jogo) - min(jogo)
    
    if not (MIN_SOMA <= soma <= MAX_SOMA): return False
    if not (MIN_PARES <= pares <= MAX_PARES): return False
    if primos < MIN_PRIMOS: return False
    if fibo < MIN_FIBO: return False
    if amplitude < MIN_AMPLITUDE: return False
    return True

if __name__ == "__main__":
    print("\n==============================================")
    print("      SISTEMA DE PREVISÃO HÍBRIDA (NÍVEL 5)    ")
    print("      IA + Estatística + Probabilidade Real    ")
    print("==============================================\n")

    try:
        if not os.path.exists(ARQUIVO_LSTM) or not os.path.exists(ARQUIVO_GRU):
            raise FileNotFoundError("Modelos não encontrados. Rode main.py!")

        print("🧠 Carregando Especialistas (LSTM & GRU)...")
        model_lstm = load_model(ARQUIVO_LSTM)
        model_gru = load_model(ARQUIVO_GRU)
        
        ultimos_jogos = carregar_ultimos_jogos()

        if len(ultimos_jogos) == WINDOW_SIZE:
            
            mlb = MultiLabelBinarizer(classes=range(1, NUM_NUMEROS + 1))
            mlb.fit([list(range(1, 61))])
            dados_numeros = mlb.transform(ultimos_jogos)
            dados_features = calcular_features_extras(ultimos_jogos)
            
            input_combined = np.hstack((dados_numeros, dados_features))
            input_reshaped = input_combined.reshape(1, WINDOW_SIZE, INPUT_DIM)

            # Previsão do Ensemble
            pred_lstm = model_lstm.predict(input_reshaped, verbose=0)[0]
            pred_gru = model_gru.predict(input_reshaped, verbose=0)[0]
            predicao_final = (pred_lstm + pred_gru) / 2.0

            # RANKING GERAL DE PROBABILIDADES
            lista_probabilidades = []
            for i in range(NUM_NUMEROS):
                numero = i + 1
                prob = predicao_final[i] * 100 
                lista_probabilidades.append((numero, prob))
            
            # Ordena do maior para o menor
            lista_probabilidades.sort(key=lambda x: x[1], reverse=True)
            
            # Seleciona as 18 dezenas mais fortes (Pool)
            indices_top = predicao_final.argsort()[-18:][::-1]
            dezenas_ouro = sorted([i + 1 for i in indices_top])
            
            print("\n🔍 Filtrando as 18 dezenas de ouro...")
            print(f"   Pool: {dezenas_ouro}")
            
            print("⚙️  Gerando combinações perfeitas e calculando probabilidades...")
            todas_combinacoes = list(combinations(dezenas_ouro, 6))
            
            jogos_rankeados = []
            for jogo in todas_combinacoes:
                if validar_jogo(jogo):
                    probs_individuais = [predicao_final[n-1] for n in jogo]
                    score_total = sum(probs_individuais)
                    confianca_media = (score_total / 6.0) * 100
                    
                    jogos_rankeados.append({
                        'numeros': jogo,
                        'confianca': confianca_media,
                        'probs': probs_individuais
                    })
            
            jogos_rankeados.sort(key=lambda x: x['confianca'], reverse=True)
            
            print(f"   Jogos aprovados pelo filtro matemático: {len(jogos_rankeados)}")

            # EXIBIÇÃO DOS RESULTADOS 
            print("\n" + "="*50)
            print("🏆  TOP 1 - MELHOR COMBINAÇÃO ESTATÍSTICA  🏆")
            print("="*50)
            
            if len(jogos_rankeados) > 0:
                melhor = jogos_rankeados[0]
                print(f"NUMEROS: {list(melhor['numeros'])}")
                print(f"⭐ Confiança da IA neste jogo: {melhor['confianca']:.2f}%")
                print("\nDetalhes (Probabilidade de cada bola sair):")
                for num, prob in zip(melhor['numeros'], melhor['probs']):
                    print(f"   Bola {num:02d}: {prob*100:.2f}%")
                
                print("-" * 50)
                if len(jogos_rankeados) > 1:
                    print(f"🥈 OPÇÃO 2: {list(jogos_rankeados[1]['numeros'])} ({jogos_rankeados[1]['confianca']:.2f}%)")
                if len(jogos_rankeados) > 2:
                    print(f"🥉 OPÇÃO 3: {list(jogos_rankeados[2]['numeros'])} ({jogos_rankeados[2]['confianca']:.2f}%)")
            else:
                print("⚠️ O filtro matemático foi muito rigoroso e eliminou todas as combinações das 18 dezenas.")
                print("   Tente rodar novamente ou relaxar os parâmetros MIN_SOMA/MAX_SOMA.")

            # EXIBIÇÃO DO RANKING DE TODOS OS NÚMEROS
            print("\n" + "#"*60)
            print("📊  RAIO-X COMPLETO: PROBABILIDADE DE CADA NÚMERO (1-60)")
            print("#"*60)
            
            print("\n🔥 ZONA QUENTE (Top 15 - Os favoritos da IA)")
            for i in range(15):
                num, prob = lista_probabilidades[i]
                print(f"   #{i+1:02d} | Dezena {num:02d}: {prob:.2f}%")
                
            print("\n😐 ZONA MORNA (Do 16 ao 45 - Médio risco)")
            # Imprime em duas colunas para economizar espaço
            for i in range(15, 45, 2):
                n1, p1 = lista_probabilidades[i]
                n2, p2 = lista_probabilidades[i+1]
                print(f"   Dezena {n1:02d}: {p1:.2f}%   |   Dezena {n2:02d}: {p2:.2f}%")

            print("\n❄️ ZONA FRIA (Bottom 15 - A IA acha que NÃO vem)")
            for i in range(45, 60):
                num, prob = lista_probabilidades[i]
                print(f"   #{i+1:02d} | Dezena {num:02d}: {prob:.2f}%")

            print("\n" + "="*60)

        else:
            print("Dados insuficientes.")

    except Exception as e:
        print(f"Erro crítico: {e}")