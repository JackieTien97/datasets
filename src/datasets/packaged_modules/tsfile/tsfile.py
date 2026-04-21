from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class TsFileConfig(datasets.BuilderConfig):
    """BuilderConfig for Apache TsFile.

    Args:
        columns (`list[str]`, *optional*):
            Explicit list of logical timeseries paths to load. When set, the
            ``devices``, ``fields`` and ``path_prefix`` filters must be left
            unset. Use ``TsFileDataFrame.list_timeseries()`` to inspect
            available series for a given file.
        devices (`list[str]`, *optional*):
            Keep only series whose tag-value segment matches one of the given
            device identifiers. Equivalent to "give me everything from these
            devices". Combined with ``fields`` and ``path_prefix`` as a logical
            AND. The match is exact and segment-based, e.g. ``devices=["d1"]``
            keeps ``mytable.d1.temperature`` but not ``mytable.d10.temperature``.
        fields (`list[str]`, *optional*):
            Keep only series whose final path segment (a.k.a. *field*, *sensor*
            or *measurement*) matches one of the given names. Equivalent to
            "give me this measurement across all devices".
        path_prefix (`str`, *optional*):
            Keep only series whose path starts with this prefix, delegated to
            ``TsFileDataFrame.list_timeseries(path_prefix=...)``. Note that
            the prefix is matched as a raw string and should not include a
            trailing dot.
        start_time (`int`, *optional*):
            Lower bound (inclusive) of the timestamp range to load.
            Defaults to the minimum int64 (no lower bound).
        end_time (`int`, *optional*):
            Upper bound (inclusive) of the timestamp range to load.
            Defaults to the maximum int64 (no upper bound).
        features (`Features`, *optional*):
            Cast the data to ``features``.
        on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error"):
            Specify what to do when a TsFile cannot be read.
    """

    columns: Optional[list[str]] = None
    devices: Optional[list[str]] = None
    fields: Optional[list[str]] = None
    path_prefix: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    features: Optional[datasets.Features] = None
    on_bad_files: Literal["error", "warn", "skip"] = "error"

    def __post_init__(self):
        super().__post_init__()
        if self.columns is not None and (
            self.devices is not None or self.fields is not None or self.path_prefix is not None
        ):
            raise ValueError(
                "`columns` cannot be combined with `devices`, `fields` or `path_prefix`. "
                "Use either an explicit list of series paths, or one or more of the "
                "device/field/prefix predicates."
            )


class TsFile(datasets.ArrowBasedBuilder):
    """Apache TsFile builder backed by ``tsfile.TsFileDataFrame``.

    Each ``.tsfile`` file is opened as a ``TsFileDataFrame`` and converted to
    a single Arrow table whose first column is ``timestamp`` (int64) followed
    by one float64 column per logical timeseries.
    """

    BUILDER_CONFIG_CLASS = TsFileConfig

    def _info(self):
        if (
            self.config.columns is not None
            and self.config.features is not None
            and set(self.config.columns) | {"timestamp"} != set(self.config.features)
        ):
            raise ValueError(
                "The columns and features arguments must reference the same series, but got "
                f"{self.config.columns} and {list(self.config.features)}"
            )
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if self.info.features is None:
                for file in files:
                    try:
                        self.info.features = datasets.Features.from_arrow_schema(self._infer_schema(file))
                        break
                    except Exception as e:
                        if self.config.on_bad_files == "error":
                            logger.error(f"Failed to read schema from '{file}' with error {type(e).__name__}: {e}")
                            raise
                        elif self.config.on_bad_files == "warn":
                            logger.warning(f"Skipping bad schema from '{file}'. {type(e).__name__}: {e}")
                        else:
                            logger.debug(f"Skipping bad schema from '{file}'. {type(e).__name__}: {e}")
            if self.info.features is None:
                raise ValueError(
                    f"At least one valid data file must be specified, all the data_files are invalid: {self.config.data_files}"
                )
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        for file_idx, file in enumerate(files):
            try:
                pa_table = self._read_file_to_arrow(file)
            except Exception as e:
                if self.config.on_bad_files == "error":
                    logger.error(f"Failed to read file '{file}' with error {type(e).__name__}: {e}")
                    raise
                elif self.config.on_bad_files == "warn":
                    logger.warning(f"Skipping bad file '{file}'. {type(e).__name__}: {e}")
                else:
                    logger.debug(f"Skipping bad file '{file}'. {type(e).__name__}: {e}")
                continue
            yield Key(file_idx, 0), self._cast_table(pa_table)

    # --- helpers -------------------------------------------------------

    def _resolve_columns(self, df) -> list[str]:
        # Explicit list takes precedence and is validated as-is.
        if self.config.columns is not None:
            available = set(df.list_timeseries())
            missing = [c for c in self.config.columns if c != "timestamp" and c not in available]
            if missing:
                raise ValueError(
                    f"Requested columns {missing} not found in TsFile. "
                    f"Use TsFileDataFrame.list_timeseries() to inspect available series."
                )
            return [c for c in self.config.columns if c != "timestamp"]

        # Predicate-based filtering. Start with the cheapest filter (path_prefix)
        # so the upstream reader can prune at the metadata level.
        if self.config.path_prefix is not None:
            candidates = df.list_timeseries(path_prefix=self.config.path_prefix)
        else:
            candidates = df.list_timeseries()

        if self.config.devices is not None:
            wanted_devices = set(self.config.devices)
            # The series path is "<table>.<tag_value_1>...<tag_value_n>.<field>".
            # We match exact segments between the first (table) and last (field).
            candidates = [series for series in candidates if wanted_devices.intersection(series.split(".")[1:-1])]

        if self.config.fields is not None:
            wanted_fields = set(self.config.fields)
            candidates = [series for series in candidates if series.rsplit(".", 1)[-1] in wanted_fields]

        if not candidates and (
            self.config.devices is not None or self.config.fields is not None or self.config.path_prefix is not None
        ):
            available = df.list_timeseries()
            raise ValueError(
                "No timeseries matched the requested filters "
                f"(devices={self.config.devices}, fields={self.config.fields}, "
                f"path_prefix={self.config.path_prefix!r}). "
                f"Available series in this file: {available}"
            )

        return candidates

    def _infer_schema(self, file: str) -> pa.Schema:
        from tsfile import TsFileDataFrame

        df = TsFileDataFrame(file, show_progress=False)
        try:
            series = self._resolve_columns(df)
        finally:
            df.close()
        fields = [pa.field("timestamp", pa.int64())]
        fields.extend(pa.field(name, pa.float64()) for name in series)
        return pa.schema(fields)

    def _read_file_to_arrow(self, file: str) -> pa.Table:
        from tsfile import TsFileDataFrame

        int64_min = np.iinfo(np.int64).min
        int64_max = np.iinfo(np.int64).max
        start = int64_min if self.config.start_time is None else int(self.config.start_time)
        end = int64_max if self.config.end_time is None else int(self.config.end_time)

        df = TsFileDataFrame(file, show_progress=False)
        try:
            series = self._resolve_columns(df)
            if not series:
                arrays = [pa.array(np.empty(0, dtype=np.int64))]
                return pa.table(arrays, names=["timestamp"])
            aligned = df.loc[start:end, series]
        finally:
            df.close()

        timestamps = np.asarray(aligned.timestamps, dtype=np.int64)
        values = np.asarray(aligned.values)
        arrays = [pa.array(timestamps)]
        for col_idx, name in enumerate(aligned.series_names):
            arrays.append(pa.array(np.asarray(values[:, col_idx], dtype=np.float64)))
        return pa.table(arrays, names=["timestamp", *aligned.series_names])
