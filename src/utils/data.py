from dataclasses import dataclass, astuple, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas

@dataclass
class Node():
    name: str
    parent: DataGroup | None = None
    
    def set_name(self, new_name: str):
        self.name = new_name

    def set_parent(self, new_parent: DataGroup):
        self.parent = new_parent

    def is_data(self):
        return isinstance(self, Data)
    
    def is_data_group(self):
        return isinstance(self, DataGroup)

@dataclass(frozen=True)
class DataContent:
    x: NDArray[np.float64]
    y: NDArray[np.float64]

    def __iter__(self):
        return iter(astuple(self))
    
@dataclass
class Data(Node):
    content: DataContent | None = None

@dataclass
class DataGroup(Node):
    data_list: list[Data] = field(default_factory=list)
    groups: list[DataGroup] = field(default_factory=list)

    def __post_init__(self):
        """
        Depois da inicialização do @dataclass, faz certeza com que todos os arquivos de dados
        tenham o grupo criado como pai. Ou seja, isso tira a necessidade de specificar o pai
        de um dado quando se é colocado no grupo a partir da instanciação de um objeto de grupo
        """
        for data in self.data_list:
            data.set_parent(self)

    def has_data(self):
        return len(self.data_list) > 0

    def add_group(self, data_group: DataGroup):
        return self.groups.append(data_group)
    
class DataParsingError(Exception):
    """Exceção lançada quando há um erro na leitura dos dados de um arquivo de dados"""
    pass

DEFAULT_ENCODING = "utf-8"

# O paradigma do código parte do pressuposto que todo dado deve possuir um grupo
class DataExtractor():
    def __init__(self, name: str, file_extension: str):
        self.name = name
        self.file_extension = file_extension

    def has_supported_extension(self, data_path: Path):
        return data_path.suffix == self.file_extension
    
    def extract_data(self, data_path: Path, recursive_search: bool = False) -> DataGroup:
        if not data_path.exists():
            raise DataParsingError(f"O caminho do arquivo, ou pasta, que será extraído, não existe: {data_path.absolute()}")

class XY_DataExtractor(DataExtractor):
    def __init__(self):
        super().__init__("XY", "txt")

    def _get_content_from_file(self, data_file_path: Path) -> DataContent:
        content = data_file_path.read_bytes()
        xy_data = content.decode(DEFAULT_ENCODING)

        # Se houver ocorrências de números com "," para separar decimais, converte para "."
        xy_data = xy_data.replace(",", ".")

        split_xy_data = xy_data.splitlines()
        axis_length = len(split_xy_data)

        if axis_length < 1:
            raise DataParsingError(f"Não há dados para se ler, com o extrator {self.name}, no arquivo: {data_file_path.absolute()}")

        x_axis = np.zeros(axis_length)
        y_axis = np.zeros(axis_length)

        for index, line in enumerate(split_xy_data):
            x, y = line.split("\u0009") # Esse é o código unicode do Tab[    ]
            
            x_axis[index] = np.float64(x)
            y_axis[index] = np.float64(y)
        
        return DataContent(x_axis, y_axis)

    def _create_data_from_file(self, data_file_path: Path) -> Data:
        data_name = data_file_path.name.removesuffix(data_file_path.suffix)
        data_content = self._get_content_from_file(data_file_path)
        data = Data(data_name, content=data_content)
        return data

    def _import_data_group(self, group_path: Path, parent: DataGroup, recursive_group_insert: bool) -> DataGroup:
        # Pega todos os descendentes diretos da pasta, seja arquivo ou outras sub-pastas
        sub_entities = sorted(group_path.glob("*"))

        # Pega todas os arquivos que são descendentes diretos da pasta e que tem a extensão suportada por esse extrator
        direct_file_paths = [file_path for file_path in sub_entities if (file_path.is_file() and (file_path.suffix[1:] == self.file_extension))]
        data_list = [self._create_data_from_file(data_file_path) for data_file_path in direct_file_paths]

        """
        O fato de eu estar instanciando o grupo já com a lista de dados, faz com que todos os dados tenham
        esse grupo como pai (ver o __post_init__ do DataGroup)
        """
        data_group = DataGroup(group_path.name, parent, data_list=data_list)
        if parent is not None:
            parent.add_group(data_group)

        if recursive_group_insert:
            sub_folder_paths = [sub_folder for sub_folder in sub_entities if sub_folder.is_dir()]

            for sub_folder_path in sub_folder_paths:
                # Só tenta vai tentar colocar o subgrupo, se a pasta que o representar não estiver vazia.
                if any(sub_folder_path.iterdir()):
                    self._import_data_group(sub_folder_path, data_group, recursive_group_insert)

        return data_group
    
    def extract_data(self, data_path, recursive_search = False) -> DataGroup:
        super().extract_data(data_path, recursive_search)

        if data_path.is_file():
            # Caso só esteja extraindo um arquivo, retornará o próprio nome do arquivo como o nome do grupo
            data = self._create_data_from_file(data_path)
            return DataGroup(data.name, data_list = [data])

        # A partir daqui, o caminho especificado é uma pasta
        return self._import_data_group(data_path, None, recursive_group_insert=recursive_search)

"""
Nesse formato, a primeiro coluna é o comprimento de onda absorvido. Já as próximas
colunas se referem aos valores de absorbância.
Essa lógica se aplica para todas as linhas, até na primeira, onde tem os nomes de
cada iteração de medição.
"""
class CSV_DataExtractor(DataExtractor):
    def __init__(self):
        super().__init__("CSV", "csv")

    def extract_data(self, data_path, recursive_search = False) -> DataGroup:
        super().extract_data(data_path, recursive_search)

        if not data_path.is_file():
            raise DataParsingError(f"O caminho para a extração de dados em {self.name} deve ser de um arquivo .{self.file_extension}.")

        try:
            data_frame = pandas.read_csv(data_path, delimiter=";", decimal=",", float_precision="round_trip")
        except pandas.errors.EmptyDataError:
            raise DataParsingError(f"Não há dados no arquivo: {data_path.absolute()}")
        except pandas.errors.ParserError:
            raise DataParsingError(f"Não foi ler os dados do arquivo: {data_path.absolute()}")
        except pandas.errors.DtypeWarning:
            raise DataParsingError(f"Os dados não estão formatados de maneira padronizada no arquivo: {data_path.absolute()}")

        rows, columns = data_frame.shape

        data_list: list[Data] = []
        """
        Começa da segunda linha e segunda coluna, já que as primeiras de cada só possuem informações de referência.
        - Linha 0 (com exceção do primeiro elemento): nomes das medições
        - Coluna 0 (com exceção do primeiro element): comprimentos de ondas
        """

        column_labels = data_frame.columns.values.tolist()

        info_index = 0
        start_index = info_index + 1

        for i in range(start_index, columns):
            x_axis = np.zeros(rows)
            y_axis = np.zeros(rows)

            for j in range(info_index, rows):
                data_x = data_frame.iat[j, info_index]
                data_y = data_frame.iat[j, i]

                # Não é preciso converter para o número do numpy porque o pandas já faz isso
                x_axis[j] = data_x
                y_axis[j] = data_y

            data_content = DataContent(x_axis, y_axis)
            data_name = column_labels[i]
            data = Data(data_name, content=data_content)
            data_list.append(data)

        group_name = data_path.name.removesuffix(data_path.suffix)

        return DataGroup(group_name, data_list=data_list) 
        
    
DATA_EXTRACTORS: dict[str, DataExtractor] = {
    "XY": XY_DataExtractor(),
    "CSV": CSV_DataExtractor(),
}

DATA_EXTENSIONS = (*map(lambda data_extractor: data_extractor.file_extension, DATA_EXTRACTORS.values()),)