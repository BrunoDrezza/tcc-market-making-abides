import subprocess
import time
import sys
import itertools

print("==================================================")
print(" Iniciando o Estudo de Ablação (Avellaneda-Stoikov)")
print("==================================================")

# Gera as 8 combinações possíveis (False/True) para as 3 camadas
flags = [False, True]
combinacoes = list(itertools.product(flags, flags, flags))

start_time_total = time.time()

for idx, (ofi, hedge, kill) in enumerate(combinacoes, 1):
    # Nomeando a pasta de log para sabermos exatamente qual versão rodou
    log_dir = f"TCC_Ablation_OFI_{ofi}_HEDGE_{hedge}_KILL_{kill}"
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Teste {idx}/8: OFI={ofi} | HEDGE={hedge} | KILL_SWITCH={kill}")
    
    # O comando base rodando no cenário de estresse (POV = 10%)
    comando = [
        sys.executable, "-u", "abides.py", 
        "-c", "rmsc03_as", 
        "-t", "ABM", 
        "-d", "20240101", 
        "-l", log_dir, 
        "-e", "-p", "0.1" 
    ]
    
    # Injetando as chaves se forem True
    if ofi: comando.append("--use-ofi")
    if hedge: comando.append("--use-hedge")
    if kill: comando.append("--use-kill-switch")
    
    try:
        subprocess.run(comando, check=True)
        print(f"[{time.strftime('%H:%M:%S')}] Teste {idx} concluído com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERRO no teste {idx}: {e}")
        break

end_time_total = time.time()
tempo_total = (end_time_total - start_time_total) / 60
print("\n==================================================")
print(f" Estudo de Ablação finalizado! Tempo total: {tempo_total:.2f} minutos.")
print(" Os 8 cenários foram salvos na pasta /log/ para análise de PnL.")
print("==================================================")