import pandas as pd
import numpy as np
import os

def load_agent_log(log_dir, agent_name="AVELLANEDA_STOIKOV_AGENT"):
    print(f"Loading agent log from: {log_dir}")
    filepath = os.path.join(log_dir, f"{agent_name}.bz2")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Agent log not found: {filepath}\nMake sure the simulation was run with log_orders enabled.")
    
    # Read the pickle directly
    df = pd.read_pickle(filepath)
    return df

def parse_as_metrics(df, starting_cash=10000000):
    """
    Extrai as métricas de alta frequência e calcula o PnL Tick-a-Tick e Retornos.
    starting_cash padrão: 100.000 USD (10.000.000 centimos).
    """
    print("Parsing AS quote metrics and computing temporal PnL...")
    
    # 1. Filtra apenas os eventos HFT
    if 'EventType' in df.columns:
        df_quotes = df[df['EventType'] == 'AS_QUOTE'].copy()
    else:
        text_col = 'Event' if 'Event' in df.columns else df.columns[-1]
        df_quotes = df[df[text_col].astype(str).str.contains("inv=")].copy()
        
    if df_quotes.empty:
        raise ValueError("Nenhum evento 'AS_QUOTE' foi encontrado no log.")

    # 2. Descobre qual é a coluna de texto
    text_col = 'Event' if 'Event' in df_quotes.columns else 'Message'

    # 3. Regex atualizada para capturar OBI e CASH
    regex = r"inv=(?P<inv>[-\d\.]+)\s+mid=(?P<mid>[-\d\.]+)\s+bid=(?P<bid>[-\d\.]+)\s+ask=(?P<ask>[-\d\.]+)\s+obi=(?P<obi>[-\d\.]+)\s+cash=(?P<cash>[-\d\.]+)"
    extracted = df_quotes[text_col].str.extract(regex)
    
    # 4. Junta tudo e converte tipos
    parsed_df = pd.concat([df_quotes, extracted], axis=1)
    parsed_df = parsed_df.dropna(subset=['inv', 'mid', 'bid', 'ask', 'obi', 'cash'])
    
    for col in ['inv', 'mid', 'bid', 'ask', 'obi', 'cash']:
        parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')

    # 5. Arruma o índice temporal
    if 'EventTime' in parsed_df.columns:
        parsed_df['EventTime'] = pd.to_datetime(parsed_df['EventTime'])
        parsed_df.set_index('EventTime', inplace=True)
        
    # 6. CÁLCULO DE PNL (MARK-TO-MARKET LIVRE DE VIÉS) EM DÓLARES
    parsed_df['PnL'] = ((parsed_df['cash'] - starting_cash) / 100.0) + (parsed_df['inv'] * (parsed_df['mid'] / 100.0))
    
    # NOVO: 7. CÁLCULO DA CURVA DE CAPITAL (Equity Curve) E RETORNOS
    starting_cash_usd = starting_cash / 100.0
    parsed_df['Equity'] = starting_cash_usd + parsed_df['PnL']
    
    # Calcula os retornos contínuos (log returns) da curva de capital para métricas de risco
    # O Pylance pode reclamar da divisão, mas estamos seguros pois a Equity nunca será <= 0 no ABIDES
    parsed_df['Returns'] = np.log(parsed_df['Equity'] / parsed_df['Equity'].shift(1)).fillna(0) # type: ignore
    
    print(f"Sucesso! {len(parsed_df)} cotações processadas com MtM dinâmico e Retornos.")
    return parsed_df[['inv', 'mid', 'bid', 'ask', 'obi', 'cash', 'PnL', 'Equity', 'Returns']]