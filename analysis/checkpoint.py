"""Persistencia incremental e retomada de varreduras experimentais.

Uma matriz completa de simulacoes leva dezenas de horas de relogio. Gravar os
resultados apenas ao final significa que qualquer interrupcao (queda de energia,
OOM, encerramento acidental do terminal) descarta todo o progresso. Este modulo
persiste cada simulacao assim que ela conclui e permite retomar de onde parou,
pulando as celulas ja concluidas.

O arquivo bruto de resultados e a fonte de verdade do que ja foi executado: uma
linha so e escrita apos parsing bem-sucedido, de modo que simulacoes que
falharam a meio caminho sao naturalmente reexecutadas na retomada.
"""

from __future__ import annotations

import csv
import os
import threading
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


class ResultCheckpoint:
    """Escritor incremental de resultados, seguro para uso concorrente.

    Parameters
    ----------
    path : str
        Caminho do CSV bruto, com uma linha por simulacao concluida.
    fieldnames : sequence of str
        Colunas do CSV. Devem coincidir com as chaves dos dicionarios gravados.
    key_fields : sequence of str
        Subconjunto de colunas que identifica univocamente uma simulacao
        (e.g. os parametros da celula mais a semente).

    Notes
    -----
    A escrita e serializada por um lock, pois os workers concluem em paralelo.
    Cada linha e seguida de `flush` e `os.fsync`, garantindo que o progresso
    sobreviva a uma terminacao abrupta do processo.
    """

    def __init__(
        self, path: str, fieldnames: Sequence[str], key_fields: Sequence[str]
    ) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.key_fields = list(key_fields)
        self._lock = threading.Lock()
        self._done: Set[Tuple[str, ...]] = set()

        self._load_existing()

    def _load_existing(self) -> None:
        """Carrega as chaves ja concluidas a partir do CSV, se existir."""
        if not os.path.exists(self.path):
            return

        with open(self.path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return
            if list(reader.fieldnames) != self.fieldnames:
                raise ValueError(
                    f"Cabecalho incompativel em {self.path}.\n"
                    f"  esperado: {self.fieldnames}\n"
                    f"  encontrado: {list(reader.fieldnames)}\n"
                    "Remova ou renomeie o arquivo para iniciar uma varredura nova."
                )
            for row in reader:
                self._done.add(self._key_from_row(row))

    def _key_from_row(self, row: Dict[str, Any]) -> Tuple[str, ...]:
        """Normaliza a chave de uma linha para comparacao textual."""
        return tuple(str(row[field]) for field in self.key_fields)

    def is_done(self, key: Dict[str, Any]) -> bool:
        """Indica se a simulacao identificada por `key` ja foi concluida.

        Parameters
        ----------
        key : dict
            Mapeamento contendo, ao menos, os campos declarados em `key_fields`.

        Returns
        -------
        bool
            True se a simulacao ja consta do arquivo de resultados.
        """
        return self._key_from_row(key) in self._done

    def completed_count(self) -> int:
        """Devolve o numero de simulacoes ja concluidas."""
        return len(self._done)

    def append(self, row: Dict[str, Any]) -> None:
        """Persiste uma simulacao concluida, de imediato e de forma duravel.

        Parameters
        ----------
        row : dict
            Resultado da simulacao, com as chaves declaradas em `fieldnames`.
        """
        with self._lock:
            exists = os.path.exists(self.path)
            with open(self.path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            self._done.add(self._key_from_row(row))

    def load_all(self) -> List[Dict[str, Any]]:
        """Le todos os resultados persistidos, para consolidacao final.

        Returns
        -------
        list of dict
            Linhas do CSV bruto, com valores numericos convertidos quando possivel.
        """
        if not os.path.exists(self.path):
            return []

        rows: List[Dict[str, Any]] = []
        with open(self.path, newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({k: _coerce(v) for k, v in row.items()})
        return rows


def _coerce(value: str) -> Any:
    """Converte um campo textual do CSV para bool, int ou float quando aplicavel."""
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def write_aggregate(
    path: str, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]
) -> None:
    """Reescreve o CSV agregado a partir das linhas consolidadas.

    Parameters
    ----------
    path : str
        Caminho do CSV agregado.
    fieldnames : sequence of str
        Colunas do arquivo.
    rows : iterable of dict
        Linhas consolidadas, uma por celula experimental.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
