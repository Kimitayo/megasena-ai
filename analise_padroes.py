import os
from itertools import combinations
from collections import Counter
import pandas as pd

# CONFIGURAÇÕES
ARQUIVO_DADOS = 'mega_sena-1.csv'

def carregar_todos_jogos():
    print(f"--- LENDO BANCO DE DADOS: {ARQUIVO_DADOS} ---")
    dataset = []
    if not os.path.exists(ARQUIVO_DADOS):
        print("Arquivo não encontrado!")
        return []

    try:
        with open(ARQUIVO_DADOS, 'r', encoding='utf-8', errors='ignore') as f:
            linhas = f.readlines()
            for linha in linhas:
                linha = linha.strip()
                if not linha: continue
                # Tratamento de erro de vírgula
                partes = linha.split(',') if ',' in linha else linha.split(';')
                nums = []
                for p in partes:
                    try:
                        n = int(p)
                        if 1 <= n <= 60: nums.append(n)
                    except: continue
                if len(nums) >= 6:
                    dataset.append(sorted(nums[-6:])) # pra ordem crescente
        
        print(f"Total de jogos analisados: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"Erro: {e}")
        return []

def minerar_padroes(jogos):
    print("\n--- INICIANDO MINERAÇÃO DE PADRÕES OCULTOS ---")
    
    total_jogos = len(jogos)
    
    # Conta quantas vezes cada número saiu sozinho
    todos_numeros = [num for jogo in jogos for num in jogo]
    freq_individual = Counter(todos_numeros)
    
    # Gera todos os pares possíveis de cada jogo e conta
    todos_pares = []
    for jogo in jogos:
        todos_pares.extend(combinations(jogo, 2))
    
    freq_pares = Counter(todos_pares)
    
    # ANÁLISE 1: OS PARES DE OURO (Top 10) 
    print("\n🔥 TOP 10 PARES MAIS FREQUENTES DA HISTÓRIA:")
    print("Esses números adoram sair de mãos dadas.")
    print("-" * 50)
    for par, qtd in freq_pares.most_common(10):
        porcentagem = (qtd / total_jogos) * 100
        print(f"Par {par}: Saiu {qtd} vezes ({porcentagem:.2f}% dos jogos)")

    # ANÁLISE 2: REGRAS DE ASSOCIAÇÃO (Se X sai, Y sai?) 
    # Fórmula: Confiança(A -> B) = Freq(A e B) / Freq(A)
    print("\n🧠 REGRAS DE ASSOCIAÇÃO FORTES (Confiança > 15%):")
    print("Leitura: 'Quando o número da esquerda sai, o da direita vem junto X% das vezes'")
    print("-" * 50)
    
    regras_encontradas = []
    
    # analisar apenas os pares que saíram pelo menos 15 vezes para ter relevância
    for par, qtd_juntos in freq_pares.items():
        if qtd_juntos < 15: continue
        
        num_a, num_b = par
        
        # Chance de B sair dado que A saiu
        prob_b_dado_a = (qtd_juntos / freq_individual[num_a]) * 100
        # Chance de A sair dado que B saiu
        prob_a_dado_b = (qtd_juntos / freq_individual[num_b]) * 100
        
        regras_encontradas.append((num_a, num_b, prob_b_dado_a))
        regras_encontradas.append((num_b, num_a, prob_a_dado_b))
    
    # Ordena pelas regras mais fortes
    regras_encontradas.sort(key=lambda x: x[2], reverse=True)
    
    for i in range(10): # Mostra top 10 regras
        a, b, conf = regras_encontradas[i]
        print(f"Se saiu {a:02d} -> Há {conf:.2f}% de chance de sair {b:02d}")

    # ANÁLISE 3: OS INIMIGOS (Pares que raramente saem) 
    print("\n❄️ PARES 'INIMIGOS' (Menor frequência histórica):")
    print("Se você jogar esses dois juntos, está jogando contra a estatística.")
    print("-" * 50)
    
    # Pega os últimos (menos comuns)
    pares_raros = freq_pares.most_common()[:-11:-1] 
    
    for par, qtd in pares_raros:
        print(f"Par {par}: Só saiu {qtd} vezes em toda a história!")

    # ANÁLISE 4: ATRASO ATUAL sobre quem está sumindo
    print("\n⏰ NÚMEROS MAIS ATRASADOS (Dorminhocos):")
    print("-" * 50)
    
    atrasos = {}
    # Inicializa todos com atraso infinito
    for n in range(1, 61): atrasos[n] = 0
    
    # Varre do último jogo para trás
    # Quando encontra o número, para de contar para ele
    numeros_encontrados = set()
    
    # Inverte a lista para começar do mais recente
    jogos_invertidos = jogos[::-1] 
    
    for i, jogo in enumerate(jogos_invertidos):
        for num in jogo:
            if num not in numeros_encontrados:
                atrasos[num] = i
                numeros_encontrados.add(num)
        
        if len(numeros_encontrados) == 60:
            break
            
    # Ordena pelos mais atrasados
    atrasados_ordenados = sorted(atrasos.items(), key=lambda x: x[1], reverse=True)
    
    for i in range(10):
        num, jogos_atraso = atrasados_ordenados[i]
        print(f"Número {num:02d}: Não sai há {jogos_atraso} concursos")

if __name__ == "__main__":
    dados = carregar_todos_jogos()
    if dados:
        minerar_padroes(dados)