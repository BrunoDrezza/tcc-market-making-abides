import pandas as pd
import numpy as np
import os
import re

def load_agent_log(log_dir, agent_name="AVELLANEDA_STOIKOV_AGENT"):
    print(f"Loading agent log from: {log_dir}")
    filepath = os.path.join(log_dir, f"{agent_name}.bz2")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent log not found: {filepath}\nMake sure the simulation was run with log_orders enabled and the agent name matches.")
    
    # Read the pickle directly
    df = pd.read_pickle(filepath)
    return df

def parse_as_metrics(df):
    print("Parsing AS quote metrics...")
    
    # 1. Filtra apenas os eventos que nós criamos
    if 'EventType' in df.columns:
        df_quotes = df[df['EventType'] == 'AS_QUOTE'].copy()
    else:
        # Fallback caso o ABIDES tenha mudado o nome da coluna
        text_col = 'Event' if 'Event' in df.columns else df.columns[-1]
        df_quotes = df[df[text_col].astype(str).str.contains("inv=")].copy()
        
    if df_quotes.empty:
        print("\n[RAIO-X DO ARQUIVO BZ2] O que realmente tem dentro do log:")
        print("\n--> Contagem de Eventos:")
        print(df['EventType'].value_counts() if 'EventType' in df.columns else "Coluna EventType não encontrada.")
        print("\n--> Últimas 10 linhas do log (onde deveria estar o final da simulação):")
        print(df.tail(10))
        raise ValueError("Nenhum evento 'AS_QUOTE' foi encontrado no log.")

    # 2. Descobre qual é a coluna de texto
    text_col = 'Event' if 'Event' in df_quotes.columns else 'Message'
    if text_col not in df_quotes.columns:
        text_col = df_quotes.columns[-1]

    # 3. Regex super flexível para extrair os números
    regex = r"inv=(?P<inv>[-\d\.]+)\s+mid=(?P<mid>[-\d\.]+)\s+bid=(?P<bid>[-\d\.]+)\s+ask=(?P<ask>[-\d\.]+)"
    extracted = df_quotes[text_col].str.extract(regex)
    
    # 4. Junta tudo e limpa
    parsed_df = pd.concat([df_quotes, extracted], axis=1)
    parsed_df = parsed_df.dropna(subset=['inv', 'mid', 'bid', 'ask'])
    
    if parsed_df.empty:
        print("\n[DEBUG] O Regex falhou. Veja como os textos estão salvos no arquivo:")
        print(df_quotes[text_col].head())
        raise ValueError("Nenhuma linha correspondeu ao padrão esperado do Regex.")

    # 5. Converte para números
    for col in ['inv', 'mid', 'bid', 'ask']:
        parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')

    # 6. Arruma o índice de tempo
    if 'EventTime' in parsed_df.columns:
        parsed_df['EventTime'] = pd.to_datetime(parsed_df['EventTime'])
        parsed_df.set_index('EventTime', inplace=True)
        
    print(f"Sucesso! {len(parsed_df)} cotações processadas.")
    return parsed_df[['inv', 'mid', 'bid', 'ask']]