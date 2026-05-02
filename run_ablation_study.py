import subprocess
import time
import sys
import itertools
import csv
import os
from analysis.parser import load_agent_log, parse_as_metrics # Importação direta

print("==================================================")
print(" Iniciando o Estudo de Ablacao (Avellaneda-Stoikov)")
print("==================================================")

csv_filename = "ablation_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Test_ID", "Use_OBI", "Use_HEDGE", "Use_KILL_SWITCH", 
        "Estoque_Maximo_Retido", "Estoque_Final", "PnL_Liquido_USD", "Status"
    ])

flags = [False, True]
combinacoes = list(itertools.product(flags, flags, flags))

start_time_total = time.time()

for idx, (obi, hedge, kill) in enumerate(combinacoes, 1):
    log_name = f"TCC_Ablation_OBI_{obi}_HEDGE_{hedge}_KILL_{kill}"
    log_dir = os.path.join("log", log_name)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Teste {idx}/8: OBI={obi} | HEDGE={hedge} | KILL_SWITCH={kill}")
    
    comando = [
        sys.executable, "-u", "abides.py", 
        "-c", "rmsc03_as", 
        "-t", "ABM", 
        "-d", "20240101", 
        "-l", log_name, 
        "-e", "-p", "0.1" # POV Agent acionado
    ]
    
    if obi: comando.append("--use-obi")
    if hedge: comando.append("--use-hedge")
    if kill: comando.append("--use-kill-switch")
    
    try:
        # Roda a simulação e força o sistema a esperar terminar
        subprocess.run(comando, capture_output=True, text=True, check=True)
        
        # 1. Carrega o log recém-gerado diretamente do disco
        df_raw = load_agent_log(log_dir)
        
        # 2. Faz o parsing e calcula o PnL temporal e as métricas físicas
        df_metrics = parse_as_metrics(df_raw, starting_cash=10000000)
        
        # 3. Extrai a matemática dura da curva
        estoque_maximo = df_metrics['inv'].abs().max()
        estoque_final = df_metrics['inv'].iloc[-1]
        pnl_final = df_metrics['PnL'].iloc[-1]
        
        status = "OK"
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([idx, obi, hedge, kill, estoque_maximo, estoque_final, round(pnl_final, 2), status])
            
        print(f"[{time.strftime('%H:%M:%S')}] Teste {idx} concluído. PnL Líquido: ${pnl_final:.2f} | Inventário Máximo: {estoque_maximo}")
        
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] FALHA CRITICA no teste {idx}: {str(e)}")
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([idx, obi, hedge, kill, "N/A", "N/A", "N/A", "CRASHED"])
        continue

end_time_total = time.time()
tempo_total = (end_time_total - start_time_total) / 60
print("\n==================================================")
print(f" Estudo de Ablacao finalizado! Tempo de execucao: {tempo_total:.2f} minutos.")
print(f" Matriz de resultados consolidada em: {csv_filename}")
print("==================================================")