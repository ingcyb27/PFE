import torch
import sys

def check_environment():
    print("=" * 50)
    print(" VERIFICATION DE L'ENVIRONNEMENT DE CALCUL")
    print("=" * 50)
    print(f"Version Python : {sys.version.split()[0]}")
    print(f"Version PyTorch: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_available}")
    
    if not cuda_available:
        print("[!] ATTENTION : Aucun GPU detecte. Verifiez les pilotes NVIDIA ou le container.")
        return
        
    num_devices = torch.cuda.device_count()
    print(f"Nombre de GPU detectes : {num_devices}")
    
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  [+] GPU {i} : {props.name} | VRAM Totale : {vram_gb:.2f} Go")
        
    # Petit test d'allocation memoire
    t = torch.randn(1000, 1000, device="cuda:0")
    print("[+] Test tensor CUDA : Succes !")
    print("=" * 50)

if __name__ == "__main__":
    check_environment()
