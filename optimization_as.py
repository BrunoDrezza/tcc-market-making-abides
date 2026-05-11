import subprocess
import itertools
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from analysis.parser import load_agent_log, parse_as_metrics

print("==================================================")
print(" Otimização PARALELA de Hiperparâmetros (AS)")
print("==================================================")

# O Grid de Elite (9 cenários)
# O Grid Cirúrgico (Buscando o PnL Positivo)
gammas = [0.25, 0.5, 1.0]
ks = [5.0, 10.0, 50.0]
combinacoes = list(itertools.product(gammas, ks))

csv_filename = "optimization_results.csv"

# Prepara o CSV
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Gamma", "K", "Max_Inventario", "PnL_USD", "Retorno_Pct"])

# Função que vai rodar em paralelo
def rodar_cenario(params):
    idx, total, gamma, k = params
    log_name = f"TCC_Opt_Gamma_{gamma}_K_{k}"
    log_dir = os.path.join("log", log_name)
    
    comando = [
        sys.executable, "-u", "abides.py", 
        "-c", "rmsc03_as", 
        "-t", "ABM", 
        "-d", "20240101", 
        "-l", log_name, 
        "--gamma", str(gamma),
        "--k", str(k)
    ]
    
    try:
        subprocess.run(comando, capture_output=True, text=True, check=True)
        
        df_raw = load_agent_log(log_dir)
        df_metrics = parse_as_metrics(df_raw, starting_cash=10000000)
        
        estoque_maximo = df_metrics['inv'].abs().max()
        pnl_final = df_metrics['PnL'].iloc[-1]
        retorno_pct = (df_metrics['Equity'].iloc[-1] / 100000.0 - 1) * 100
        
        return (gamma, k, estoque_maximo, pnl_final, retorno_pct, None)
        
    except subprocess.CalledProcessError as e:
        return (gamma, k, None, None, None, f"ERRO ABIDES: {e.stderr}")
    except Exception as e:
        return (gamma, k, None, None, None, f"ERRO PARSER: {e}")

if __name__ == "__main__":
    start_time = time.time()
    tarefas = [(i, len(combinacoes), g, k) for i, (g, k) in enumerate(combinacoes, 1)]
    
    work_force = 3
    print(f"Iniciando {len(combinacoes)} simulações usando {work_force} processos paralelos...\n")
    
    with ThreadPoolExecutor(max_workers=work_force) as executor:
        futuros = {executor.submit(rodar_cenario, t): t for t in tarefas}
        
        concluidos = 0
        for futuro in as_completed(futuros):
            concluidos += 1
            gamma, k, estoque_maximo, pnl_final, retorno_pct, erro = futuro.result()
            
            if erro is None and pnl_final is not None and retorno_pct is not None:
                # Forçamos o tipo para o Pylance ficar tranquilo
                pnl_val = float(pnl_final)
                ret_val = float(retorno_pct)
                
                # Salva no CSV assim que termina
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([gamma, k, estoque_maximo, round(pnl_val, 2), round(ret_val, 4)])
                print(f"[{concluidos}/{len(combinacoes)}] SUCESSO | Gamma: {gamma} | K: {k} -> PnL: ${pnl_val:.2f} | Inv. Max: {estoque_maximo}")
            else:
                print(f"[{concluidos}/{len(combinacoes)}] FALHA | Gamma: {gamma} | K: {k} -> {erro}")

    elapsed = time.time() - start_time
    print(f"\nOtimização Concluída em {elapsed:.1f} segundos! Veja o arquivo {csv_filename}.")