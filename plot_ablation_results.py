import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from analysis.parser import load_agent_log, parse_as_metrics

# =====================================================================
# 1. BLOCO DE UTILIDADES (Leitura de Dados)
# =====================================================================
def carregar_dados_ablacao(csv_path="ablation_results.csv"):
    """Lê a matriz de resultados e converte cêntimos para dólares reais."""
    if not os.path.exists(csv_path):
        print(f"[Erro] O ficheiro {csv_path} não foi encontrado.")
        return None
    
    df = pd.read_csv(csv_path)
    # Evita erros se houver CRASHED (onde os valores não são numéricos)
    df_clean = df[df['Status'] == 'OK'].copy()
    df_clean['PnL_Liquido_Cents'] = pd.to_numeric(df_clean['PnL_Liquido_Cents'])
    df_clean['PnL_Liquido_USD'] = df_clean['PnL_Liquido_Cents'] / 100.0
    
    # Cria uma label legível para o eixo X
    labels = []
    for _, row in df_clean.iterrows():
        if not row['Use_OFI'] and not row['Use_HEDGE'] and not row['Use_KILL_SWITCH']:
            labels.append("Baseline\n(A-S Clássico)")
        elif not row['Use_OFI'] and row['Use_HEDGE'] and row['Use_KILL_SWITCH']:
            labels.append("Defesa Campeã\n(Hedge+Kill)")
        else:
            labels.append(f"OFI:{str(row['Use_OFI'])[0]} | H:{str(row['Use_HEDGE'])[0]} | K:{str(row['Use_KILL_SWITCH'])[0]}")
    df_clean['Label'] = labels
    return df_clean

# =====================================================================
# 2. BLOCO DE GERAÇÃO DE GRÁFICOS
# =====================================================================
def plot_pnl_bar_chart(df, output_dir):
    """(A Fotografia Financeira) Plota o PnL Líquido lado a lado."""
    if df is None or df.empty: 
        return
    
    plt.figure(figsize=(12, 7))
    plt.title("Estudo de Ablação: Impacto no PnL Líquido sob Estresse Direcional", fontsize=16, fontweight='bold')
    
    # Define as cores (Baseline=Vermelho, Campeã=Azul, Resto=Cinzento/Laranja)
    cores = []
    for label in df['Label']:
        if "Baseline" in label: 
            cores.append('#d62728')
        elif "Campeã" in label: 
            cores.append('#1f77b4')
        elif "OFI:T" in label: 
            cores.append('#ff7f0e') # OFI piora o cenário
        else: 
            cores.append('#7f7f7f')

    bars = plt.bar(df['Label'], df['PnL_Liquido_USD'], color=cores, edgecolor='black', alpha=0.85)
    
    # Formatação do eixo Y para dólares ($)
    formatter = ticker.StrMethodFormatter('${x:,.0f}')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.axhline(0, color='black', linewidth=1.5)
    plt.ylabel("PnL Líquido Final (USD)", fontsize=12)
    
    # Adiciona os valores em cima de cada barra
    for bar in bars:
        yval = bar.get_height()
        # Posiciona o texto ligeiramente abaixo se for negativo
        offset = -1500 if yval < 0 else 1500
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, 
                 f'${yval:,.0f}', ha='center', va='bottom' if yval > 0 else 'top', 
                 fontsize=10, fontweight='bold', color='black')

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    out_path = os.path.join(output_dir, "ablation_pnl_bars.png")
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Gráfico de Barras (PnL) guardado em: {out_path}")


def plot_inventory_time_series(output_dir):
    """(O Filme do Risco) Plota a evolução física suavizada."""
    flags = [False, True]
    combinacoes = list(itertools.product(flags, flags, flags))
    
    plt.figure(figsize=(14, 8))
    plt.title("Estudo de Ablação: Evolução do Inventário sob Choque Direcional (POV=10%)", fontsize=16, fontweight='bold')
    plt.xlabel("Evolução do Dia (Nº de Cotações da Estratégia)", fontsize=12)
    plt.ylabel("Inventário Físico (Ações)", fontsize=12)

    legendas_adicionadas = []

    for ofi, hedge, kill in combinacoes:
        pasta_nome = f"TCC_Ablation_OFI_{ofi}_HEDGE_{hedge}_KILL_{kill}"
        log_dir = os.path.join("log", pasta_nome)
        
        if not os.path.exists(log_dir): 
            continue
            
        try:
            # Reutiliza o parser base silenciosamente
            raw_df = load_agent_log(log_dir)
            parsed_df = parse_as_metrics(raw_df)
            if parsed_df.empty: 
                continue
            
            # Aplica a Suavização (Média Móvel de 50)
            serie_suavizada = pd.Series(parsed_df['inv'].values).rolling(window=50, min_periods=1).mean()
            
            # Hierarquia Visual
            if not ofi and not hedge and not kill:
                cor, esp, est, label, zo, alp = '#d62728', 3.5, '-', 'Baseline (A-S 2008 Clássico)', 10, 1.0
            elif not ofi and hedge and kill:
                cor, esp, est, label, zo, alp = '#1f77b4', 3.5, '-', 'Defesa Campeã (Hedge + Kill Switch)', 10, 1.0
            elif ofi:
                label = 'Com OFI (Agrava o Risco)' if 'Com OFI' not in legendas_adicionadas else "_nolegend_"
                if label != "_nolegend_": 
                    legendas_adicionadas.append('Com OFI')
                cor, esp, est, zo, alp = '#ff7f0e', 1.5, '--', 5, 0.6
            else:
                label = 'Defesas Parciais Isoladas' if 'Parciais' not in legendas_adicionadas else "_nolegend_"
                if label != "_nolegend_": 
                    legendas_adicionadas.append('Parciais')
                cor, esp, est, zo, alp = '#7f7f7f', 1.5, ':', 4, 0.6

            # Plota usando o index sequencial (Alinhamento perfeito do Eixo X)
            plt.plot(range(len(serie_suavizada)), serie_suavizada, 
                     color=cor, linewidth=esp, linestyle=est, label=label, alpha=alp, zorder=zo)

        except Exception:
            pass # Ignora erros silenciosamente para não sujar o terminal

    plt.axhline(0, color='black', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "ablation_inventory_series.png")
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Gráfico de Série Temporal (Inventário) guardado em: {out_path}")


# =====================================================================
# 3. ORQUESTRADOR CENTRAL
# =====================================================================
def main():
    print("==================================================")
    print(" Gerando Relatório Visual Institucional (TCC)")
    print("==================================================")
    
    out_dir = "analysis_output"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Carrega os dados processados da madrugada
    df_resultados = carregar_dados_ablacao()
    
    # 2. Gera o Gráfico de Barras do PnL
    plot_pnl_bar_chart(df_resultados, out_dir)
    
    # 3. Gera a Série Temporal Suavizada do Inventário
    print("\nMinerando logs intradiários (Isto pode levar alguns segundos)...")
    plot_inventory_time_series(out_dir)
    
    print("==================================================")
    print(" Processo concluído com sucesso!")
    print(" Verifique a pasta /analysis_output/")
    print("==================================================")

if __name__ == "__main__":
    main()