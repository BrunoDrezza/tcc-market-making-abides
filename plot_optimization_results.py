import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

print("Lendo e consolidando todos os CSVs de otimização...")

# Puxa todos os arquivos CSV no diretório atual que começam com 'optimization' ou tenham 'results'
# Você pode alterar o padrão "*.csv" se tiver guardado em uma pasta específica.
arquivos_csv = glob.glob("*optimization_results*.csv") 
# Se você renomeou os arquivos, coloque os nomes exatos aqui nesta lista:
# arquivos_csv = ["grid_1.csv", "grid_2.csv", "grid_3.csv"] 

lista_df = []
for arquivo in arquivos_csv:
    try:
        df_temp = pd.read_csv(arquivo)
        lista_df.append(df_temp)
        print(f"Lido: {arquivo}")
    except Exception as e:
        print(f"Erro ao ler {arquivo}: {e}")

if not lista_df:
    print("Nenhum arquivo CSV encontrado! Verifique os nomes e pastas.")
    exit()

# Funde todos os dados em um único DataFrame
df_consolidado = pd.concat(lista_df, ignore_index=True)

# Remove duplicatas baseadas em Gamma e K (mantém o teste mais recente)
df_consolidado = df_consolidado.drop_duplicates(subset=["Gamma", "K"], keep="last")

# Ordena os valores para garantir que a matriz fique sequencial
df_consolidado = df_consolidado.sort_values(by=["Gamma", "K"])

# Criando a matriz pivô para o Heatmap
pivot_df = df_consolidado.pivot(index="Gamma", columns="K", values="PnL_USD")

# Configuração de estilo acadêmico
#sns.set_theme(style="white")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Criação da Figura (Ajuste o figsize se o grid ficar muito grande)
plt.figure(figsize=(10, 8))

# Usando a paleta 'RdYlGn' (Red-Yellow-Green) onde lucros ficam Verdes e prejuízos Vermelhos!
ax = sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                 cbar_kws={'label': 'PnL Acumulado (USD)'}, linewidths=.5)

plt.title("Mapa de Calor Consolidado: Otimização de $\gamma$ e $k$", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Aversão ao Risco ($\gamma$)", fontsize=12, fontweight='bold')
plt.xlabel("Sensibilidade de Execução ($k$)", fontsize=12, fontweight='bold')

# Inverter o eixo Y para o Gamma maior ficar em cima
ax.invert_yaxis()

plt.tight_layout()

os.makedirs("figuras", exist_ok=True)
caminho = "figuras/heatmap_consolidado.png"
plt.savefig(caminho, dpi=300, bbox_inches="tight")
print(f"\nSucesso absoluto! Heatmap salvo em: {caminho}")