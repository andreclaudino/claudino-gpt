import os
import glob
from claudino_gpt.configurations.training_configuration import TrainingConfiguration
import torch
from typing import Optional, Tuple
from claudino_gpt.model.claudino_gpt import ClaudinoGPT


def save_training_state(
    model: ClaudinoGPT,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    configuration: TrainingConfiguration,
    tokens_seen: int,
) -> str:
    checkpoints_dir = os.path.join(configuration.output_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{global_step}.pth")
    
    torch.save({
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    _cleanup_old_checkpoints(checkpoints_dir, configuration.total_models_to_keep)

    return checkpoint_path


def _cleanup_old_checkpoints(checkpoints_dir: str, total_to_keep: int):
    """
    Remove os checkpoints mais antigos, mantendo apenas os `total_to_keep` mais recentes.
    Assume que os arquivos seguem o padrão: checkpoint_<step>.pth
    """
    if total_to_keep <= 0:
        return

    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_*.pth"))
    if len(checkpoint_files) <= total_to_keep:
        return

    # Extrai o step de cada arquivo e ordena por step (decrescente = mais recente primeiro)
    def extract_step(filepath: str) -> int:
        basename = os.path.basename(filepath)
        # Remove "checkpoint_" e ".pth", depois converte para int
        try:
            return int(basename.replace("checkpoint_", "").replace(".pth", ""))
        except ValueError:
            return -1  # Ignora arquivos mal formatados

    checkpoint_files.sort(key=extract_step, reverse=True)
    files_to_delete = checkpoint_files[total_to_keep:]

    for f in files_to_delete:
        try:
            os.remove(f)
        except OSError:
            pass  # Ignora erros de remoção


def load_latest_training_state(
    model: ClaudinoGPT,
    optimizer: torch.optim.Optimizer,
    configuration,
    device: str
) -> Tuple[Optional[int], Optional[int], bool]:
    """
    Tenta carregar o checkpoint mais recente. Se falhar, tenta o próximo mais recente,
    até que todos os checkpoints tenham sido testados. Retorna (global_step, tokens_seen, success).
    """
    checkpoints_dir = os.path.join(configuration.output_path, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        return None, None, False

    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_*.pth"))
    if not checkpoint_files:
        return None, None, False

    # Ordena do mais recente (maior step) para o mais antigo
    def extract_step(filepath: str) -> int:
        basename = os.path.basename(filepath)
        try:
            return int(basename.replace("checkpoint_", "").replace(".pth", ""))
        except ValueError:
            return -1  # Ignora arquivos mal formatados

    checkpoint_files.sort(key=extract_step, reverse=True)

    # Tenta carregar cada checkpoint, do mais recente ao mais antigo
    for checkpoint_path in checkpoint_files:
        try:
            print(f"Tentando carregar checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Validação mínima: garantir que as chaves essenciais existam
            if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint:
                print(f"⚠️  Checkpoint {checkpoint_path} está incompleto. Pulando.")
                continue

            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            global_step = checkpoint.get('global_step', None)
            tokens_seen = checkpoint.get('tokens_seen', 0)

            if global_step is None:
                print(f"⚠️  Checkpoint {checkpoint_path} não contém 'global_step'. Pulando.")
                continue

            print(f"✅ Checkpoint carregado com sucesso: {checkpoint_path}")
            return global_step, tokens_seen, True

        except (RuntimeError, OSError, EOFError, ValueError, KeyError) as e:
            print(f"❌ Falha ao carregar {checkpoint_path}: {e}. Tentando próximo checkpoint...")
            os.remove(checkpoint_path)
            print(f"🗑️  Checkpoint inválido removido: {checkpoint_path}")
            continue

    # Nenhum checkpoint foi carregado com sucesso
    print("🚫 Nenhum checkpoint válido encontrado.")
    return None, None, False


def convert_checkpoint_to_inference_model(
    checkpoint_path: str,
    output_model_path: str,
    device: str
):
    """
    Converte um checkpoint de treinamento em um modelo de inferência (apenas state_dict do modelo).
    Salva apenas os pesos do modelo, pronto para carregar com `model.load_state_dict()`.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model_state_dict']
    torch.save(model_state, output_model_path)
    print(f"Modelo de inferência salvo em: {output_model_path}")