from pathlib import Path
from typing import Annotated

import typer
from utils.data import DATA_EXTRACTORS

import matplotlib.pyplot as plt

def get_absorbance_data(data_file: Path):
    if not data_file.is_file():
        print("O caminho para o arquivo de dados de ser um arquivo.")
        return

app = typer.Typer()

@app.command()
def show_graph(
    data_path: Annotated[
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
    transmittance: bool = False
):
    data_extractor = DATA_EXTRACTORS["CSV"]
    main_data_group = data_extractor.extract_data(data_path)

    cmap = plt.get_cmap('coolwarm')

    num_plots = len(main_data_group.data_list)

    for index, data in enumerate(main_data_group.data_list):
        color_index = index / (num_plots - 1) # Normalizado
        line_color = cmap(color_index)

        x, y = data.content

        # Convertendo para transmitância, se escolhido.
        # Relação entre absorbância e transmitância: A = -log(T)
        if transmittance:
            y = 10**(-y)
        
        plt.plot(x, y, label = data.name, color=line_color)

    """
    Como a absorbância "explode" para mais de 1, e os dados depois dessa medida são irrelevantes
    para a análise, nós vamos limitar. Como a transmitância não passa de 1, por conta da relação
    logarítmica, não precisamos limitar para ela.
    """
    if not transmittance:
        plt.ylim(0, 1)
    
    quantity_label = "Transmitância" if transmittance else "Absorbância"
    plt.get_current_fig_manager().set_window_title(f"Gráfico de {quantity_label}")
    plt.title(f"Comprimento de onda vs {quantity_label}")
    plt.xlabel("Comprimento de onda")
    plt.ylabel(quantity_label)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()