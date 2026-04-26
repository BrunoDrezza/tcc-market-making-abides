import subprocess
import time
import sys
import itertools
import csv
import re

print("==================================================")
print(" Iniciando o Estudo de Ablacao (Avellaneda-Stoikov)")
print("==================================================")

# Prepara o ficheiro CSV e escreve o cabecalho institucional
csv_filename = "ablation_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Test_ID", "Use_OFI", "Use_HEDGE", "Use_KILL_SWITCH", 
        "Estoque_Final", "Final_Cash_Cents", "Marked_To_Market_Cents", 
        "PnL_Liquido_Cents", "Status"
    ])

# Gera as 8 combinacoes possiveis (False/True) para a matriz do estudo
flags = [False, True]
combinacoes = list(itertools.product(flags, flags, flags))

start_time_total = time.time()

for idx, (ofi, hedge, kill) in enumerate(combinacoes, 1):
    log_name = f"TCC_Ablation_OFI_{ofi}_HEDGE_{hedge}_KILL_{kill}"
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Teste {idx}/8: OFI={ofi} | HEDGE={hedge} | KILL_SWITCH={kill}")
    
    comando = [
        sys.executable, "-u", "abides.py", 
        "-c", "rmsc03_as", 
        "-t", "ABM", 
        "-d", "20240101", 
        "-l", log_name, 
        "-e", "-p", "0.1" 
    ]
    
    if ofi: 
        comando.append("--use-ofi")
    if hedge: 
        comando.append("--use-hedge")
    if kill: 
        comando.append("--use-kill-switch")
    
    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True)
        
        # Regex para extracao do inventario, caixa e MtM a partir do stdout
        holdings_match = re.search(r"Final holdings for AVELLANEDA_STOIKOV_AGENT:\s*\{.*?ABM:\s*([-\d]+).*?CASH:\s*([-\d]+)\s*\}.*?Marked to market:\s*([-\d]+)", resultado.stdout)
        
        # Regex para extracao do PnL Liquido do bloco de agregacao final
        pnl_match = re.search(r"AvellanedaStoikovAgent:\s*([-\d]+)", resultado.stdout)
        
        estoque_final = holdings_match.group(1) if holdings_match else "ERRO"
        final_cash = holdings_match.group(2) if holdings_match else "ERRO"
        mtm = holdings_match.group(3) if holdings_match else "ERRO"
        pnl_liquido = pnl_match.group(1) if pnl_match else "ERRO"
        
        status = "OK" if (holdings_match and pnl_match) else "PARSE_ERROR"
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([idx, ofi, hedge, kill, estoque_final, final_cash, mtm, pnl_liquido, status])
            
        print(f"[{time.strftime('%H:%M:%S')}] Teste {idx} concluido. PnL Liquido: {pnl_liquido} cêntimos.")
        
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] FALHA CRITICA no teste {idx}.")
        # Em caso de crash, regista a falha no CSV e continua a execucao
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([idx, ofi, hedge, kill, "N/A", "N/A", "N/A", "N/A", "CRASHED"])
        continue

end_time_total = time.time()
tempo_total = (end_time_total - start_time_total) / 60
print("\n==================================================")
print(f" Estudo de Ablacao finalizado! Tempo de execucao: {tempo_total:.2f} minutos.")
print(f" Matriz de resultados consolidada em: {csv_filename}")
print("==================================================")