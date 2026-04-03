import subprocess
import time
import sys  # <--- Importação necessária para travar o ambiente

pov_levels = [
    ("01", "0.01"),
    ("05", "0.05"),
    ("10", "0.1"),
    ("50", "0.5")
]

print("==================================================")
print(" Iniciando a Bateria de Testes de Impacto (POV)")
print("==================================================")

start_time_total = time.time()

for label, pov in pov_levels:
    log_dir = f"TCC_POV_{label}"
    print(f"\n[{time.strftime('%H:%M:%S')}] A iniciar simulação para POV = {pov} ({label}%)")
    print(f"Os resultados serão guardados em: log/{log_dir}")
    
    # O SEGREDO ESTÁ AQUI: sys.executable garante que ele usa o ABIDES_Enviroment
    comando = [
        sys.executable, "-u", "abides.py", 
        "-c", "rmsc03", 
        "-t", "ABM", 
        "-d", "20240101", 
        "-l", log_dir, 
        "-e", "-p", pov
    ]
    
    try:
        subprocess.run(comando, check=True)
        print(f"[{time.strftime('%H:%M:%S')}] Simulação POV {label}% concluída com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERRO na simulação POV {label}%: {e}")
        print("A interromper a fila de testes.")
        break

end_time_total = time.time()
tempo_total = (end_time_total - start_time_total) / 60
print("\n==================================================")
print(f" Todas as simulações concluídas! Tempo total: {tempo_total:.2f} minutos.")
print(" Pode agora gerar o gráfico de impacto no mid-price.")
print("==================================================")