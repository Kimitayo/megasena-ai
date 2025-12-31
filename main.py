import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os

# CONFIGURAÇÕES
ARQUIVO_DADOS = 'mega_sena-1.csv'
ARQUIVO_LSTM = 'cerebro_lstm.keras'
ARQUIVO_GRU = 'cerebro_gru.keras'
WINDOW_SIZE = 20
NUM_NUMEROS = 60

NUM_FEATURES_EXTRAS = 10 
INPUT_DIM = NUM_NUMEROS + NUM_FEATURES_EXTRAS 

# Parte Estatística
PRIMOS = set([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
FIBONACCI = set([1, 2, 3, 5, 8, 13, 21, 34, 55])

def carregar_dados():
    print(f"Lendo arquivo {ARQUIVO_DADOS}")
    dataset_final = []
    
    if not os.path.exists(ARQUIVO_DADOS):
        print("Arquivo não encontrado!")
        return []

    try:
        with open(ARQUIVO_DADOS, 'r', encoding='utf-8', errors='ignore') as f:
            linhas = f.readlines()
            for linha in linhas:
                linha = linha.strip()
                if not linha: continue
                # Tratamento 
                partes = linha.split(',') if ',' in linha else linha.split(';')
                
                numeros_nesta_linha = []
                for pedaco in partes:
                    try:
                        num = int(pedaco)
                        if 1 <= num <= 60: numeros_nesta_linha.append(num)
                    except ValueError: continue
                
                # Garante que pegamos apenas os 6 números do sorteio
                if len(numeros_nesta_linha) >= 6:
                    jogo = sorted(numeros_nesta_linha[-6:])
                    if len(set(jogo)) == 6: dataset_final.append(jogo)
        
        print(f"Sucesso! {len(dataset_final)} jogos carregados.")
        return dataset_final
    except Exception as e:
        print(f"Erro crítico ao ler CSV: {e}")
        return []

def calcular_features_extras(jogos):
    """Gera as 10 estatísticas avançadas para cada jogo"""
    features = []
    max_soma = 345 
    
    for i in range(len(jogos)):
        jogo = jogos[i]
        
        qtd_pares = sum(1 for n in jogo if n % 2 == 0)
        soma = sum(jogo)
        qtd_primos = sum(1 for n in jogo if n in PRIMOS)
        qtd_fibo = sum(1 for n in jogo if n in FIBONACCI)
        
        # Compara com o anterior da lista
        if i > 0:
            repetidos = sum(1 for n in jogo if n in set(jogos[i-1]))
        else:
            repetidos = 0 
            
        # 6-9: Quadrantes do bilhete
        q1 = q2 = q3 = q4 = 0
        for n in jogo:
            if n <= 30:
                if n % 10 != 0 and (n % 10) <= 5: q1 += 1
                else: q2 += 1
            else:
                if n % 10 != 0 and (n % 10) <= 5: q3 += 1
                else: q4 += 1
        
        # 10: Amplitude
        amplitude = max(jogo) - min(jogo)
        
        # Normalização (0.0 a 1.0) para a Rede Neural gostar
        features.append([
            qtd_pares/6.0, soma/max_soma, qtd_primos/6.0, qtd_fibo/6.0, repetidos/6.0,
            q1/6.0, q2/6.0, q3/6.0, q4/6.0, amplitude/59.0
        ])
        
    return np.array(features)

# ARQUITETURAS NEURAIS
def criar_modelo_lstm(input_shape):
    model = Sequential(name="LSTM_Expert")
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_NUMEROS, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def criar_modelo_gru(input_shape):
    model = Sequential(name="GRU_Expert")
    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_NUMEROS, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("--- TREINAMENTO DE ENSEMBLE (NÍVEL 5 - ANTIOVERFITTING) ---")
    
    jogos = carregar_dados()
    
    if len(jogos) > WINDOW_SIZE:
        print("Processando inteligência estatística...")
        
        # BLINDAGEM
        mlb = MultiLabelBinarizer(classes=range(1, NUM_NUMEROS + 1))
        dados_numeros = mlb.fit_transform(jogos)
        
        dados_features = calcular_features_extras(jogos)
        
        # Fusão dos dados
        dados_completos = np.hstack((dados_numeros, dados_features))
        print(f"Matriz de Treino: {dados_completos.shape}")

        # Criação das Janelas Temporais
        X, y = [], []
        for i in range(WINDOW_SIZE, len(dados_completos)):
            X.append(dados_completos[i-WINDOW_SIZE:i]) # Passado 
            y.append(dados_numeros[i]) # Futuro 
            
        X, y = np.array(X), np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        
        # TREINO 1: LSTM
        print("\n>>> Treinando Especialista LSTM...")
        model_lstm = criar_modelo_lstm((WINDOW_SIZE, INPUT_DIM))
        
        # Agora olhar para 'val_loss'
        ckpt_lstm = ModelCheckpoint(ARQUIVO_LSTM, monitor='val_loss', save_best_only=True, verbose=1)
        early = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
        
        model_lstm.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_test, y_test), callbacks=[ckpt_lstm, early], verbose=1)
        
        # TREINO 2: GRU
        print("\n>>> Treinando Especialista GRU...")
        model_gru = criar_modelo_gru((WINDOW_SIZE, INPUT_DIM))
        
        # Agora olhar para 'val_loss'
        ckpt_gru = ModelCheckpoint(ARQUIVO_GRU, monitor='val_loss', save_best_only=True, verbose=1)
        
        model_gru.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_test, y_test), callbacks=[ckpt_gru, early], verbose=1)

        print("\n--- SISTEMA ATUALIZADO COM SUCESSO ---")
        print("Os modelos agora estão calibrados para generalizar e não apenas decorar.")
    else:
        print("Erro: CSV com poucos dados.")