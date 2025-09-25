import json
from pathlib import Path


def write_metric_to_file(data_dict, file_path):
    """
    Escreve um dicionário como uma nova linha em um arquivo.
    
    Args:
        data_dict (dict): Dicionário a ser escrito no arquivo
        file_path (str): Caminho para o arquivo
    
    Returns:
        bool: True se a operação foi bem-sucedida, False caso contrário
    """
    try:
        # Converte o dicionário para string JSON
        dict_string = json.dumps(data_dict)
        
        # Cria o caminho até o arquivo se necessário
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Abre o arquivo em modo append e escreve a nova linha
        with open(file_path, 'a') as file:
            file.write(dict_string + '\n')
        
        return True
    
    except Exception as e:
        print(f"Erro ao escrever no arquivo: {e}")
        return False
