from typing import Annotated, Literal
from pathlib import Path

import typer
import matplotlib.pyplot as plt
from utils.data import DATA_EXTRACTORS

app = typer.Typer()

GRAPH_LABELS = {
    "emiss": "Emissão",
    "exc": "Excitação",
    "sync": "Síncrona"
}

@app.command()
def show_graph(
    data_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
        )
    ],
    graph_type: Literal["emiss", "exc", "sync"] = "emiss"
):
    data_extractor = DATA_EXTRACTORS["XY"]
    main_data_group = data_extractor.extract_data(data_path)

    cmap = plt.get_cmap('coolwarm')

    num_plots = len(main_data_group.data_list)

    for index, data in enumerate(main_data_group.data_list):
        color_index = index / (num_plots - 1) # Normalizado
        line_color = cmap(color_index)

        x, y = data.content

        # Convertendo para transmitância, se escolhido.
        # Relação entre absorbância e transmitância: A = -log(T)
        
        plt.plot(x, y, label = data.name, color=line_color)

    """
    Como a absorbância "explode" para mais de 1, e os dados depois dessa medida são irrelevantes
    para a análise, nós vamos limitar. Como a transmitância não passa de 1, por conta da relação
    logarítmica, não precisamos limitar para ela.
    """

    plt.get_current_fig_manager().set_window_title(f"Gráfico de Espectrofluorimetria {GRAPH_LABELS[graph_type]}")
    plt.title(f"Comprimento de onda vs Intensidade ({GRAPH_LABELS[graph_type]})")
    plt.xlabel("Comprimento de onda")
    plt.ylabel("Intensidade")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    app()