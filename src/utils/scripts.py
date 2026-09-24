from pathlib import Path
from typing import Annotated
import codecs

import typer
import cchardet as chardet

app = typer.Typer()

@app.command()
def check_encoding(
    file_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
):

    if file_path.stat().st_size == 0:
        print("O arquivo está vazio, não é possível identificar a codificação.")
        return

    # Só lê os primeiros 50KB do arquivo para não carregar todo o arquivo na memória desnecessariamente 
    with file_path.open('rb') as f:
        raw_data = f.read(51200)

    encoding = chardet.detect(raw_data)["encoding"]
    confidence = chardet.detect(raw_data)["confidence"]
    print(f"A codificação do arquivo é: {encoding}. Com uma confiança de: {confidence*100:.2f}%.")

@app.command()
def convert_encoding(
    file_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    target: str,
    old: str = None,
):
    try:
        codecs.lookup(target)
    except(LookupError):
        print(f"A codificação de destino {target} não existe.")
        return

    if old is not None:        
        try:
            codecs.lookup(target)
        except(LookupError):
            print(f"A codificação de referência {old} não existe.")
            return

    if file_path.stat().st_size == 0:
        print("O arquivo está vazio, não é possível realizar a conversão.")
        return

    current_encoding = old

    if current_encoding is None:
        # Só lê os primeiros 50KB do arquivo para não carregar todo o arquivo na memória desnecessariamente 
        with file_path.open('rb') as f:
            raw_data = f.read(51200)
        
        current_encoding = chardet.detect(raw_data)["encoding"]

    if current_encoding is None:
        print("Não foi possível achar a codificação atual do arquivo.")
        return

    print(f"Convertendo o arquivo de {current_encoding} para {target}...")

    # Agora lê a pasta toda para converter todo o arquivo
    raw_data = file_path.read_bytes()
    decoded_raw_data = raw_data.decode(current_encoding)

    previous_name = file_path.name.removesuffix(file_path.suffix)
    extension = file_path.suffix
    new_name = f"{previous_name}_{target}{extension}"

    new_encoded_file_path = file_path.with_name(new_name)
    new_encoded_file_path.write_text(decoded_raw_data, encoding=target)

    print("Arquivo convertido com sucesso!")
    
    

if __name__ == "__main__":
    app()