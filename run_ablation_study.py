import subprocess
import time
import sys
import itertools
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from analysis.parser import load_agent_log, parse_as_metrics

print("==================================================")
print(" Iniciando Estudo de Ablacao PARALELIZADO (AS)")
print("==================================================")

csv_filename = "ablation_results.csv"

# Prepara o cabeçalho do CSV
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Test_ID", "Use_OBI", "Use_HEDGE", "Use_KILL_SWITCH", 
        "Estoque_Maximo_Retido", "Estoque_Final", "PnL_Liquido_USD", "Retorno_Total_Pct", "Status"
    ])

flags = [False, True]
combinacoes = list(itertools.product(flags, flags, flags))

# Função Worker que roda cada cenário
def rodar_cenario_ablation(params):
    idx, obi, hedge, kill = params
    log_name = f"TCC_Ablation_OBI_{obi}_HEDGE_{hedge}_KILL_{kill}"
    log_dir = os.path.join("log", log_name)
    
    comando = [
        sys.executable, "-u", "abides.py", 
        "-c", "rmsc03_as", 
        "-t", "ABM", 
        "-d", "20240101", 
        "-s", "20240101",
        "-l", log_name, 
        "-e", "-p", "0.25"  
    ]
    
    if obi: comando.append("--use-obi")
    if hedge: comando.append("--use-hedge")
    if kill: comando.append("--use-kill-switch")
    
    try:
        subprocess.run(comando, capture_output=True, text=True, check=True)
        
        df_raw = load_agent_log(log_dir)
        starting_capital_cents = 10000000
        df_metrics = parse_as_metrics(df_raw, starting_cash=starting_capital_cents)
        
        estoque_maximo = df_metrics['inv'].abs().max()
        estoque_final = df_metrics['inv'].iloc[-1]
        pnl_final = df_metrics['PnL'].iloc[-1]
        
        capital_inicial_usd = starting_capital_cents / 100.0
        retorno_total_pct = (df_metrics['Equity'].iloc[-1] / capital_inicial_usd - 1) * 100
        
        return (idx, obi, hedge, kill, estoque_maximo, estoque_final, pnl_final, retorno_total_pct, "OK", None)
        
    except subprocess.CalledProcessError as e:
        return (idx, obi, hedge, kill, "N/A", "N/A", "N/A", "N/A", "CRASHED", f"ERRO ABIDES: {e.stderr}")
    except Exception as e:
        return (idx, obi, hedge, kill, "N/A", "N/A", "N/A", "N/A", "CRASHED", f"ERRO PARSER: {e}")

if __name__ == "__main__":
    start_time_total = time.time()
    
    tarefas = [(idx, obi, hedge, kill) for idx, (obi, hedge, kill) in enumerate(combinacoes, 1)]
    work_force = 3 # Usa 3 núcleos, deixa 1 pro Windows/Word
    
    print(f"Disparando 8 cenarios com POV em {work_force} processos simultaneos...\n")
    
    with ThreadPoolExecutor(max_workers=work_force) as executor:
        futuros = {executor.submit(rodar_cenario_ablation, t): t for t in tarefas}
        
        concluidos = 0
        for futuro in as_completed(futuros):
            concluidos += 1
            idx, obi, hedge, kill, estoque_maximo, estoque_final, pnl_final, retorno_total_pct, status, erro = futuro.result()
            
            if erro is None and pnl_final != "N/A":
                # Força o float para o Pylance e salva
                pnl_val = float(pnl_final)
                ret_val = float(retorno_total_pct)
                
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([idx, obi, hedge, kill, estoque_maximo, estoque_final, round(pnl_val, 2), round(ret_val, 4), status])
                
                print(f"[{concluidos}/8] OK | Teste {idx} (OBI={obi}, HEDGE={hedge}, KILL={kill}) -> PnL: ${pnl_val:.2f} | Inv. Max: {estoque_maximo}")
            else:
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([idx, obi, hedge, kill, "N/A", "N/A", "N/A", "N/A", "CRASHED"])
                print(f"[{concluidos}/8] FALHA | Teste {idx} -> {erro}")

    tempo_total = (time.time() - start_time_total) / 60
    print("\n==================================================")
    print(f" Estudo de Ablacao PARALELO finalizado! Tempo: {tempo_total:.2f} min.")
    print(f" Matriz de resultados em: {csv_filename}")
    print("==================================================")