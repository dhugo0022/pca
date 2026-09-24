from pathlib import Path

DATA_EXTENSIONS = "txt"

def truncate_lines_on_file(cut_length: float, cut_path: Path):
    sub_entities = list(cut_path.glob("*"))

    direct_file_paths = [file_path for file_path in sub_entities if (file_path.is_file() and (file_path.suffix[1:] in DATA_EXTENSIONS))]

    for file_path in direct_file_paths:
        file_text = file_path.read_text("utf-8")

        file_text = file_text.replace(",", ".")

        split_xy_data = file_text.splitlines()

        cut_index = 0
        for index, line in enumerate(split_xy_data):
            x, _ = line.split("\u0009") # Esse código unicode é do Tab[    ]

            if float(x) == cut_length:
                cut_index = index + 1
                break
        
        truncated_xy_data = split_xy_data[:cut_index]
        truncated_file_text = "\n".join(truncated_xy_data)

        file_path.write_text(truncated_file_text)

    sub_folder_paths = [sub_folder for sub_folder in sub_entities if sub_folder.is_dir()]

    [truncate_lines_on_file(cut_length, sub_folder_path) for sub_folder_path in sub_folder_paths]
  
def main():
    cut_path = Path(input("Digite o caminho da pasta com os arquivos para truncar: "))
    cut_length = int(input("Valor de comprimento de onda máximo: "))
    truncate_lines_on_file(cut_length, cut_path)
    print("Truncamento realizado com sucesso!")

if __name__ == "__main__":
    main()