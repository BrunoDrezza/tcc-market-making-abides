import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.parser import load_agent_log, parse_as_metrics

# Configuração de estilo acadêmico
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 10})

def plot_ablation_comparison():
    print("Iniciando a geração dos gráficos comparativos da Ablação...")
    
    flags = [False, True]
    combinacoes = list(itertools.product(flags, flags, flags))
    
    # Criar a figura com 2 subplots (PnL em cima, Inventário embaixo)
    # Criar a figura com 2 subplots (PnL em cima, Inventário embaixo)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax_pnl = axes[0] # Subplot superior
    ax_inv = axes[1] # Subplot inferior
    
    # Paleta de cores para os 8 cenários (destacando o clássico e os melhores)
    colors = sns.color_palette("tab10", 8)
    
    for idx, (obi, hedge, kill) in enumerate(combinacoes):
        log_name = f"TCC_Ablation_OBI_{obi}_HEDGE_{hedge}_KILL_{kill}"
        log_dir = os.path.join("log", log_name)
        
        # Cria um rótulo legível para o gráfico
        label = f"OBI:{'ON' if obi else 'OFF'} | Hedge:{'ON' if hedge else 'OFF'} | Kill:{'ON' if kill else 'OFF'}"
        
        # Destacar o modelo Clássico (tudo OFF) em vermelho e o Completo (tudo ON) em azul forte
        linewidth = 1.5
        if not obi and not hedge and not kill:
            color = 'red'
            linewidth = 2.5
            label = "Clássico (Tudo OFF)"
        elif obi and hedge and kill:
            color = 'blue'
            linewidth = 2.5
            label = "Modificado (Tudo ON)"
        else:
            color = colors[idx]
        
        try:
            # Carrega e converte os dados usando o seu parser blindado
            df_raw = load_agent_log(log_dir)
            df_metrics = parse_as_metrics(df_raw, starting_cash=10000000)
            
            # Subplot 1: Curva de PnL (Mark-to-Market em USD)
            ax_pnl.plot(df_metrics.index, df_metrics['PnL'], label=label, color=color, linewidth=linewidth)
            
            # Subplot 2: Dinâmica de Inventário
            ax_inv.plot(df_metrics.index, df_metrics['inv'], color=color, linewidth=linewidth, alpha=0.7)
            
        except Exception as e:
            print(f"Aviso: Não foi possível plotar o cenário {label}. Erro: {e}")
            continue

    # Formatação do Gráfico de PnL
    ax_pnl.set_title("Estudo de Ablação: Evolução do Profit & Loss (PnL) Marcado a Mercado", fontsize=14, fontweight='bold')
    ax_pnl.set_ylabel("PnL Acumulado (USD)", fontsize=12)
    ax_pnl.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_pnl.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Formatação do Gráfico de Inventário
    ax_inv.set_title("Dinâmica de Inventário sob Risco Direcional", fontsize=14, fontweight='bold')
    ax_inv.set_ylabel("Inventário (Ações)", fontsize=12)
    ax_inv.set_xlabel("Tempo de Simulação", fontsize=12)
    ax_inv.axhline(0, color='black', linestyle='--', linewidth=1)
    
    # Ajusta o layout para a legenda não cortar
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    
    # Salvar a imagem
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "ablation_pnl_inventory.png")
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"\nGráfico comparativo salvo com sucesso em: {filepath}")

if __name__ == "__main__":
    plot_ablation_comparison()