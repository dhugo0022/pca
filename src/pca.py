from pathlib import Path
from typing import Callable, Annotated
import copy
from io import StringIO
import math

from tabulate import tabulate
import typer
import matplotlib.pyplot as plt

import scipy as sp
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from utils.data import Data, DataGroup, DataContent, DATA_EXTRACTORS

def process_group_data(data_group: DataGroup, process: Callable[[Data], Data], recursive: bool = False) -> DataGroup:
    processed_data_list = [process(data) for data in data_group.data_list]
    data_group.data_list = processed_data_list

    if recursive:
        [process_group_data(child_group, process, recursive) for child_group in data_group.groups]

    return data_group

def save_data_group_disk(data_group: DataGroup, directory: Path, recursive: bool = False) -> Path:
    group_path = directory.joinpath(data_group.name)

    if group_path.exists():
        print("Já existe uma pasta como destino para salvamento. Delete antes de salvar com esse nome")
        return group_path
    else:
        group_path.mkdir(parents=True)

    for data in data_group.data_list:
        data_path = group_path.joinpath(f"{data.name}.txt")

        if not data_path.exists():
            data_path.touch()

        str_buffer = StringIO()

        data_content = data.content
        for i in range(0, data_content.x.size):
            x = data_content.x[i]
            y = data_content.y[i]
            str_buffer.write(f"{x:.4f}\u0009{y:.4f}\n") # Tab: \u0009

        data_path.write_text(str_buffer.getvalue())
        str_buffer.close()

    if recursive:
        [save_data_group_disk(child_group, group_path, recursive) for child_group in data_group.groups]

    return group_path

def get_data_process_and_save(data_path: Path, process_name: str, process: Callable[[Data], Data]) -> Path:
    main_data_group = DATA_EXTRACTORS["XY"].extract_data(data_path, recursive_search=True)

    processed_data_group = process_group_data(copy.deepcopy(main_data_group), process, True)
    processed_data_group.set_name(f"{main_data_group.name}_{process_name}")

    return save_data_group_disk(processed_data_group, data_path.parent, True)

"""
data_directory: nome do diretório contendo pastas que separam as concentrações diferentes

Detalhe 1: os arquivos pegos serão aqueles imediatos a pasta, ou seja, não serão pego arquivos
de sub-pastas dessa pasta (não é uma pesquisa recursiva).

Detalhe 2: os nomes das pastas deverão estar padronizados no formato: "export{concentração}%"

Detalhe 3: os nomes dos arquivos deverão estar padronizados no formato: "exec{comprimento de onda}nm.txt"
- "exc" -> Para indicar que é um espectro de excitação

Cria uma matriz onde, em cada coluna, há um comprimento diferente e, em cada linha, há concentração
diferente.
"""
def get_group_from_data_directory(data_path: Path) -> DataGroup:
    data_extractor = DATA_EXTRACTORS["XY"]
    data_group = data_extractor.extract_data(data_path, recursive_search=True)
    return data_group


X_START_OFFSET = 10
X_END_BOUND = 680
X_STEP = 0.5

"""
Cria várias matrizes cuja as linhas são populadas por concentrações e as colunas pelo espectro de emissão 
obtidas pelas iterações de excitação por um certo comprimento de onda
# Exemplo:
- Concentração (20, 40, 60, 80, 10) nas linhas e os valores de intensidade 
"""
def build_concentration_exc_matrices(data_path: Path) -> tuple[list[float], list[DataGroup], list[NDArray]]:
    data_group = get_group_from_data_directory(data_path)

    # Pega os comprimentos de excitação e as concentrações e organiza elas de forma crescente
    excitations = sorted([float(data.name[3:][:-2]) for data in data_group.groups[0].data_list])
    concentrations = sorted(data_group.groups, key=lambda group: float(group.name[6:][:-1]))

    matrices = []

    # Itera para cada excitação de excitação
    for excitation in excitations:
        """
        Essa lista é que vai ser usada como "blueprint" para a array de concentração vs intensidade.
        A estrutura dessa lista será a seguinte:
        - Nas linhas: as concentrações
        - Nas colunas: as intensidades dos respectivos comprimentos de onda
        """
        concentration_intensity_list = []
        # print(f"Criando spectro para a excitação: {excitation} nm.")

        # Cada grupo significa uma concentração diferente
        for contentration_data_group in concentrations:
            # concentration = float(group.name[6:][:-1]) 
            # print(f"Criando grupo de dados para a concentração: {concentration}.")

            # Essa lista vai ser populada por valores de intensidade dos respectivos comprimentos de onda
            concentration_spectrum = []

            for data in contentration_data_group.data_list:
                data_excitation = float(data.name[3:][:-2])
                
                # Limite a análise para a excitação em foco
                if data_excitation != excitation:
                    continue

                """
                Trunca as arrays para que todas fiquem do mesmo tamanho, só assim o método PCA pode ser realizado.
                Neste caso, limita que o índice máximo seja lastreado no comprimento máximo de onda de emissão de 680 nm
                """
                list_new_length = len(data.content.x[data.content.x <= X_END_BOUND]) 
                trucated_intensity_list = data.content.y[:list_new_length]
                concentration_spectrum = trucated_intensity_list.astype(float).tolist()

            # print(f"Tamanho concetration_spectrum: {len(concentration_spectrum)}.")
            
            concentration_intensity_list.append(concentration_spectrum)

        # print(f"Tamanho spectra: {len(concentration_intensity_list)}.")
        
        X = np.array(concentration_intensity_list)
        # print(f"Matrix shape: {X.shape}")
        matrices.append(X)

    return excitations, concentrations, matrices


def print_pca_info(data_path: Path, target_excitation: float):
    excitations, concentrations, exc_matrices = build_concentration_exc_matrices(data_path)
    concentration_percentages = [float(concentration.name[6:][:-1]) for concentration in concentrations]

    for index in range(0, len(excitations)):
        excitation = excitations[index]
        if excitation != target_excitation:
            continue

        excitation_matrix = exc_matrices[index]
        
        pca = PCA()
        """
        Retorna uma matriz de forma (n_amostras x n_componentes).

        Nessa matriz cada linha é uma concentração e cada coluna é a coordenada daquela
        concentração naquela componente.

        scores = escores
        """
        scores = pca.fit_transform(excitation_matrix)

        """
        Retorna uma matriz de forma (n_componentes x n_comprimentos de onda).

        Cada linha é uma componente principal; cada coluna é o peso daquele comprimento de onda de 
        emissão naquela componente. Ou seja, a matriz diz quanto cada comprimento de onda contribui 
        para formar cada componente."

        loadings = pesos
        """
        loadings = pca.components_
        # print(f"Loadings shape: {loadings.shape}")

        """
        Retorna uma lista com o valor, em porcentagem normalizada, de quanto cada comportante 
        é responsável por provocar a mudança nos valores analizados.
        """
        explained_percentage = pca.explained_variance_ratio_ * 100 # Convertendo para escala percentual
        # print(f"Excitation: {excitation}")

        # Um step é adicionado no limite final do arranjo para que o valor final seja incluido
        wavelengths = np.arange(excitation + X_START_OFFSET, X_END_BOUND + X_STEP, X_STEP)
        # print(f"Wavelengths shape: {wavelengths.shape}")

        excitation_figure, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Gráfico de Peso vs Comprimento de onda
        scr_vs_wavelength_axis = axes[0]

        # Gráfico de Escore vs Concentração
        scr_vs_concentration_axis = axes[1]

        for pc_index, pci_loading in enumerate(loadings):
            pc_label = f"PC{pc_index + 1} - {explained_percentage[pc_index]:.2f}%"
            scr_vs_wavelength_axis.plot(wavelengths, pci_loading, label=pc_label, alpha = 1 / (pc_index +  1))

        scr_vs_wavelength_axis.set_title(f"Peso vs Comprimento de onda de emissão")
        scr_vs_wavelength_axis.set_ylabel("Peso")
        scr_vs_wavelength_axis.set_xlabel(f"Comprimento de onda de emissão em {int(excitation)} nm")
        scr_vs_wavelength_axis.legend(loc="upper right")

        # Aqui tem que iterar por coluna, já que os valores da componente estão lá
        for pc_index in range(scores.shape[1]):
            scr_vs_concentration_axis.scatter(concentration_percentages, scores[:, pc_index], label=f"PC{pc_index + 1}")
            scr_vs_concentration_axis.plot(concentration_percentages, scores[:, pc_index], alpha=0.4)  # linha mais fraca

        scr_vs_concentration_axis.set_title(f"Escore vs Concentração")
        scr_vs_concentration_axis.set_ylabel("Escore")
        scr_vs_concentration_axis.set_xlabel("Concentração (%)")
        scr_vs_concentration_axis.legend(loc="upper right")

        plt.tight_layout()
        plt.ioff()
        plt.show()


def print_tabulated_pca_info(data_path: Path):
    excitations, _, exc_matrices = build_concentration_exc_matrices(data_path)

    pca_results = []

    for index, _ in enumerate(excitations):
        excitation_matrix = exc_matrices[index]
        pca = PCA()
        pca.fit_transform(excitation_matrix)

        explained = pca.explained_variance_ratio_ * 100

        pca_results.append(explained)

    
    # Monta a tabela
    table = []

    for pc in range(5):
        row = [f"PC{pc + 1}"]

        for result in pca_results:
            row.append(result[pc])

        table.append(row)

    # PC1 + PC2
    table.append([
        "PC1 + PC2",
        *[result[0] + result[1] for result in pca_results]
    ])

    headers = ["PC"] + [f"{exc:.0f} nm" for exc in excitations]

    print(
        tabulate(
            table,
            headers=headers,
            floatfmt=".2f",
            tablefmt="simple"
        )
    )

def smoothing_process(old_data: Data):
        old_data_content = old_data.content

        # Linearização
        convolution_width = 15 # Tamanho da janela de análise (o tanto de pontos considerado ao redor de um outro pontos) de suavização/derivada
        polyorder = 3 # Grau do polinômio que vai ser utilizado para aproximar e depois derivar
        deriv = 0 # Ordem da derivada
        delta=0.5 # Espaçamento entre medidas

        y_smooth = sp.signal.savgol_filter(
            old_data_content.y, 
            window_length=convolution_width, 
            polyorder=polyorder, 
            deriv = deriv, # Quando a derivada é zero, ele vai só suavizar
            delta=delta
        )

        return Data(old_data.name, old_data.parent, DataContent(old_data_content.x, y_smooth))

def second_derivative_process(old_data: Data):
        old_data_content = old_data.content

        # Linearizaração
        convolution_width = 15 # Tamanho da janela de análise (o tanto de pontos considerado ao redor de um outro pontos) de suavização/derivada
        polyorder = 3 # Grau do polinômio que vai ser utilizado para aproximar e depois derivar
        deriv = 2 # Ordem da derivada
        delta=0.5 # Espaçamento entre medidas

        y_smooth = sp.signal.savgol_filter(
            old_data_content.y, 
            window_length=convolution_width, 
            polyorder=polyorder, 
            deriv = deriv,
            delta=delta
        )

        return Data(old_data.name, old_data.parent, DataContent(old_data_content.x, y_smooth))

app = typer.Typer()

@app.command()
def main(data_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
        )
    ]
):
    # Passa a segunda derivada em todos os dados e lineariza ao mesmo tempo utilizando o filtro de Savitzky-Golay
    smoothed_data_path = get_data_process_and_save(data_path, "suavizado", smoothing_process)
    second_derivative_data_path = get_data_process_and_save(data_path, "segunda_derivada", second_derivative_process)

    print_pca_info(data_path, 380)
    print_pca_info(smoothed_data_path, 380)
    
    print_pca_info(data_path, 400)
    
    print_pca_info(second_derivative_data_path, 440)
    print_pca_info(second_derivative_data_path, 350)



   
if __name__ == "__main__":
    app()