from dataclasses import dataclass
from typing import Iterator, Literal, Optional

import pandas as pd
import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


def _tsdatatype_to_arrow(ts_dtype, *, timestamp_unit: str = "ms", timestamp_tz: Optional[str] = None) -> pa.DataType:
    """Map a tsfile ``TSDataType`` to its closest Arrow type."""
    from tsfile.constants import TSDataType

    mapping = {
        TSDataType.BOOLEAN: pa.bool_(),
        TSDataType.INT32: pa.int32(),
        TSDataType.INT64: pa.int64(),
        TSDataType.TIMESTAMP: pa.timestamp(timestamp_unit, tz=timestamp_tz),
        TSDataType.FLOAT: pa.float32(),
        TSDataType.DOUBLE: pa.float64(),
        TSDataType.TEXT: pa.string(),
        TSDataType.STRING: pa.string(),
        TSDataType.DATE: pa.date32(),
        TSDataType.BLOB: pa.binary(),
    }
    return mapping.get(ts_dtype, pa.string())


def _to_epoch_int(value: pa.TimestampScalar, unit: str) -> int:
    """Convert a PyArrow timestamp scalar to an integer epoch in the given unit."""
    return value.cast(pa.timestamp(unit)).value


def _promote_tsdatatype(a, b):
    """Return the wider of two ``TSDataType`` values.

    IoTDB supports ``ALTER COLUMN ... SET DATA TYPE`` with the following
    promotion paths (see ``TypeInferenceUtils.canAutoCast``):

    - INT32 → INT64, FLOAT, DOUBLE
    - INT64 → DOUBLE
    - FLOAT → DOUBLE

    Before compaction rewrites old TsFiles, the same column may legitimately
    carry different types across files.  This helper picks the wider type so
    that every file's data can be safely cast into the merged schema.
    """
    if a == b:
        return a

    from tsfile.constants import TSDataType

    # Explicit promotion table: (a, b) → result.
    # The graph is a DAG, NOT a linear chain:
    #   INT32 → INT64 → DOUBLE
    #   INT32 → FLOAT → DOUBLE
    # INT64 and FLOAT are incompatible with each other; their common
    # supertype is DOUBLE.
    _PROMOTE = {
        (TSDataType.INT32, TSDataType.INT64): TSDataType.INT64,
        (TSDataType.INT32, TSDataType.FLOAT): TSDataType.FLOAT,
        (TSDataType.INT32, TSDataType.DOUBLE): TSDataType.DOUBLE,
        (TSDataType.INT64, TSDataType.FLOAT): TSDataType.DOUBLE,
        (TSDataType.INT64, TSDataType.DOUBLE): TSDataType.DOUBLE,
        (TSDataType.FLOAT, TSDataType.DOUBLE): TSDataType.DOUBLE,
    }
    pair = (a, b)
    if pair in _PROMOTE:
        return _PROMOTE[pair]
    pair = (b, a)
    if pair in _PROMOTE:
        return _PROMOTE[pair]

    # Non-numeric or unrelated types: cannot promote.
    raise ValueError(
        f"Incompatible column types across files: {a.name} vs {b.name}. "
        f"Only numeric widening (INT32→INT64→DOUBLE, INT32→FLOAT→DOUBLE) is supported."
    )


@dataclass
class TsFileConfig(datasets.BuilderConfig):
    """BuilderConfig for Apache TsFile (table model only).

    Args:
        table_name (`str`, *optional*):
            Name of the table to read. When unset, the first table found in the
            first valid file is used. Lookups are case-insensitive.
        columns (`list[str]`, *optional*):
            Subset of columns (tag and/or field names) to read from the table.
            When unset, all columns of the table are returned. Columns that are
            absent from a particular file are filled with nulls.
        start_time (`pa.TimestampScalar`, *optional*):
            Inclusive lower bound for the timestamp range, as a PyArrow
            timestamp scalar (e.g. ``pa.scalar(value, type=pa.timestamp("ms"))``
            ). Defaults to no lower bound.
        end_time (`pa.TimestampScalar`, *optional*):
            Inclusive upper bound for the timestamp range. Same type as
            ``start_time``. Defaults to no upper bound.
        batch_size (`int`, *optional*, defaults to 100_000):
            Maximum number of rows per Arrow record batch. Larger values reduce
            per-batch overhead at the cost of more memory.
        features (`Features`, *optional*):
            Final Features schema. When provided, schema inference (and the
            associated metadata scan over input files) is skipped entirely.
        on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error"):
            How to react when a file cannot be opened or does not contain the
            requested table.
        timestamp_unit (`str`, *optional*, defaults to "ms"):
            Time unit used for the timestamp column. Must be one of
            ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``.  IoTDB defaults to
            milliseconds; change this if the source database was configured
            differently.
        timestamp_tz (`str`, *optional*):
            Time zone for the timestamp column (e.g. ``"UTC"``,
            ``"Asia/Shanghai"``). When unset, the timestamp is timezone-naive.
    """

    table_name: Optional[str] = None
    columns: Optional[list[str]] = None
    start_time: Optional[pa.TimestampScalar] = None
    end_time: Optional[pa.TimestampScalar] = None
    batch_size: int = 100_000
    features: Optional[datasets.Features] = None
    on_bad_files: Literal["error", "warn", "skip"] = "error"
    timestamp_unit: Literal["s", "ms", "us", "ns"] = "ms"
    timestamp_tz: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be a positive integer, got {self.batch_size}")
        if self.columns is not None and len(self.columns) == 0:
            raise ValueError("`columns` must be a non-empty list when provided.")
        if self.timestamp_unit not in ("s", "ms", "us", "ns"):
            raise ValueError(f"`timestamp_unit` must be one of 's', 'ms', 'us', 'ns', got {self.timestamp_unit!r}")
        if self.on_bad_files not in ("error", "warn", "skip"):
            raise ValueError(f"`on_bad_files` must be one of 'error', 'warn', 'skip', got {self.on_bad_files!r}")
        if self.start_time is not None:
            self.start_time = _to_epoch_int(self.start_time, self.timestamp_unit)
        if self.end_time is not None:
            self.end_time = _to_epoch_int(self.end_time, self.timestamp_unit)


class TsFile(datasets.ArrowBasedBuilder):
    """Apache TsFile builder (table model).

    Each input file is read through ``tsfile.to_dataframe`` in streaming mode and
    emitted as Arrow record batches that follow the union of the table schema
    seen across all files (IoTDB schema evolution: columns absent from a given
    file are filled with nulls).
    """

    BUILDER_CONFIG_CLASS = TsFileConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_table_name: Optional[str] = None
        self._requested_columns: Optional[list[str]] = None

    def _info(self):
        if (
            self.config.columns is not None
            and self.config.features is not None
            and set(self.config.columns) != set(self.config.features)
        ):
            raise ValueError(
                "The columns and features argument must contain the same columns, but got "
                f"{self.config.columns} and {self.config.features}"
            )
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download(self.config.data_files)

        # Lowercase user inputs to match tsfile's case-insensitive convention.
        self._resolved_table_name: Optional[str] = self.config.table_name.lower() if self.config.table_name else None
        self._requested_columns: Optional[list[str]] = (
            [c.lower() for c in self.config.columns] if self.config.columns else None
        )

        splits = []
        if self.info.features is None:
            # TsFile columns can differ across files (IoTDB schema evolution),
            # so we must scan files from ALL splits to build a complete union
            # schema — unlike Parquet where every file carries the full column set.
            all_files = [f for file_list in data_files.values() for f in file_list]
            resolved_table, features = self._infer_features(all_files)
            if features is None:
                raise ValueError(
                    "Could not infer schema from any of the provided files. "
                    "Set `features` explicitly or check the input files."
                )
            self._resolved_table_name = resolved_table
            self.info.features = features

        if self._resolved_table_name is None:
            # User pinned `features` but not `table_name`: still need a table to query.
            all_files = [f for file_list in data_files.values() for f in file_list]
            self._resolved_table_name = self._discover_first_table(all_files)

        for split_name, files in data_files.items():
            files = list(files)
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_shards(self, files):
        # Per-file sharding. Splitting further (e.g. per device/tag combination)
        # could be added later if a single file becomes the bottleneck.
        yield from files

    def _generate_tables(self, files):
        target_schema = self.info.features.arrow_schema
        for file_idx, file in enumerate(files):
            try:
                yield from self._iter_file(file_idx, file, target_schema)
            except _SkipFile:
                continue

    # --- schema inference ---------------------------------------------

    def _infer_features(self, files):
        """Return ``(resolved_table_name, Features)`` from the input files.

        All files are always scanned so that type promotion (e.g.
        INT32→INT64→DOUBLE) across schema-evolved files is handled correctly.
        Columns that are requested but never appear in any file fall back to
        ``float64`` and are filled with nulls at read time.
        """
        from tsfile.constants import TIME_COLUMN, ColumnCategory

        wanted_table = self._resolved_table_name
        requested_columns = self._requested_columns

        merged_columns: dict = {}  # name -> tsfile.ColumnSchema (lazy import)
        time_column_name: Optional[str] = None
        resolved_table: Optional[str] = wanted_table
        wanted_set = set(requested_columns) if requested_columns is not None else None

        for file in files:
            try:
                with self._open_reader(file) as reader:
                    schemas = reader.get_all_table_schemas()
                    self._require_table_model(file, schemas)
                    if resolved_table is None:
                        resolved_table = next(iter(schemas))
                    if resolved_table not in schemas:
                        # File doesn't have the requested table -> skip per `on_bad_files`.
                        raise _MissingTableError(resolved_table, list(schemas))
                    table_schema = schemas[resolved_table]
                    for col in table_schema.get_columns():
                        name = col.get_column_name()
                        if col.get_category() == ColumnCategory.TIME:
                            time_column_name = name
                            continue
                        if wanted_set is not None and name not in wanted_set:
                            continue
                        existing = merged_columns.get(name)
                        if existing is None:
                            merged_columns[name] = col
                        else:
                            wider = _promote_tsdatatype(existing.get_data_type(), col.get_data_type())
                            if wider != existing.get_data_type():
                                merged_columns[name] = col
            except Exception as e:
                if self._should_reraise(file, e):
                    raise
                continue

        if resolved_table is None:
            return None, None

        time_field_name = time_column_name or TIME_COLUMN
        return resolved_table, self._build_features(time_field_name, merged_columns, requested_columns)

    def _build_features(
        self,
        time_field_name: str,
        merged_columns: dict,
        requested_columns: Optional[list[str]],
    ) -> datasets.Features:
        """Assemble the Arrow schema and wrap it in ``Features``."""
        from tsfile.constants import ColumnCategory

        ts_arrow_kwargs = {"timestamp_unit": self.config.timestamp_unit, "timestamp_tz": self.config.timestamp_tz}
        fields: list[pa.Field] = [
            pa.field(time_field_name, pa.timestamp(self.config.timestamp_unit, tz=self.config.timestamp_tz))
        ]

        if requested_columns is not None:
            # User specified columns — preserve user-requested order. Use the
            # discovered tsfile column type when available; fall back to float64
            # for columns absent from every input file (filled with nulls later).
            for name in requested_columns:
                if name == time_field_name:
                    continue
                col = merged_columns.get(name)
                if col is not None:
                    fields.append(pa.field(name, _tsdatatype_to_arrow(col.get_data_type(), **ts_arrow_kwargs)))
                else:
                    fields.append(pa.field(name, pa.float64()))
        else:
            # No columns specified — use merged schema.
            # Ordering: tags first, then fields, each in insertion order.
            tag_cols = [c for c in merged_columns.values() if c.get_category() == ColumnCategory.TAG]
            field_cols = [c for c in merged_columns.values() if c.get_category() == ColumnCategory.FIELD]
            for col in (*tag_cols, *field_cols):
                fields.append(
                    pa.field(col.get_column_name(), _tsdatatype_to_arrow(col.get_data_type(), **ts_arrow_kwargs))
                )

        # Drop duplicate-named fields just in case (e.g. user listed the time column).
        seen: set[str] = set()
        deduped = []
        for f in fields:
            if f.name in seen:
                continue
            seen.add(f.name)
            deduped.append(f)
        return datasets.Features.from_arrow_schema(pa.schema(deduped))

    def _discover_first_table(self, files) -> Optional[str]:
        """Return the first table name found in the first valid file."""
        for file in files:
            try:
                with self._open_reader(file) as reader:
                    schemas = reader.get_all_table_schemas()
                    self._require_table_model(file, schemas)
                    return next(iter(schemas))
            except Exception as e:
                if self._should_reraise(file, e):
                    raise
                continue
        return None

    # --- per-file streaming -------------------------------------------

    def _iter_file(self, file_idx: int, file: str, target_schema: pa.Schema):
        """Yield ``(Key, pa.Table)`` tuples for one file or skip on failure."""
        try:
            available = self._available_columns_in_file(file)
        except Exception as e:
            if self._should_reraise(file, e):
                raise
            raise _SkipFile from None

        # Restrict the column projection to whatever this file actually has.
        target_names = list(target_schema.names)
        if self._requested_columns is not None:
            requested_present = [c for c in self._requested_columns if c in available]
            columns_to_read = requested_present
        else:
            columns_to_read = None

        try:
            iterator = self._iter_chunks(file, columns_to_read)
        except Exception as e:
            if self._should_reraise(file, e):
                raise
            raise _SkipFile from None

        try:
            for batch_idx, df in enumerate(iterator):
                pa_table = self._dataframe_to_arrow(df, target_schema, target_names)
                yield Key(file_idx, batch_idx), pa_table
        except Exception as e:
            if self._should_reraise(file, e):
                raise
            raise _SkipFile from None

    def _iter_chunks(self, file: str, columns_to_read: Optional[list[str]]) -> Iterator[pd.DataFrame]:
        from tsfile import to_dataframe

        return to_dataframe(
            file_path=file,
            table_name=self._resolved_table_name,
            column_names=columns_to_read,
            start_time=self.config.start_time,
            end_time=self.config.end_time,
            max_row_num=self.config.batch_size,
            as_iterator=True,
        )

    def _available_columns_in_file(self, file: str) -> set[str]:
        """Return the lowercased set of non-time column names in our table."""
        from tsfile.constants import ColumnCategory

        with self._open_reader(file) as reader:
            schemas = reader.get_all_table_schemas()
            self._require_table_model(file, schemas)
            if self._resolved_table_name not in schemas:
                raise _MissingTableError(self._resolved_table_name, list(schemas))
            table_schema = schemas[self._resolved_table_name]
            return {
                col.get_column_name()
                for col in table_schema.get_columns()
                if col.get_category() != ColumnCategory.TIME
            }

    def _dataframe_to_arrow(self, df: pd.DataFrame, target_schema: pa.Schema, target_names: list[str]) -> pa.Table:
        """Align a chunk DataFrame to the target Arrow schema.

        Missing columns are filled with typed nulls; extra columns are dropped.
        """
        lowered = {col: col.lower() for col in df.columns}
        if len(set(lowered.values())) < len(lowered):
            dupes = [c for c in lowered.values() if list(lowered.values()).count(c) > 1]
            raise ValueError(
                f"Column name conflict after case-folding: {sorted(set(dupes))}. "
                "DataFrame contains columns that differ only by case."
            )
        df = df.rename(columns=lowered)
        n_rows = len(df)
        arrays: list[pa.Array] = []
        for field in target_schema:
            if field.name in df.columns:
                arrays.append(pa.array(df[field.name], type=field.type, from_pandas=True))
            else:
                arrays.append(pa.nulls(n_rows, type=field.type))
        pa_table = pa.Table.from_arrays(arrays, names=target_names)
        return table_cast(pa_table, target_schema)

    # --- error handling -----------------------------------------------

    @staticmethod
    def _open_reader(file: str):
        from tsfile import TsFileReader

        # Guard against corrupt files: the C library's TsFileReader constructor
        # silently returns an invalid handle (instead of raising) when the file
        # is not a valid TsFile, and any subsequent call on it segfaults.
        # Pre-check the 6-byte magic header to avoid the crash.
        _TSFILE_MAGIC = b"TsFile"
        try:
            with open(file, "rb") as f:
                header = f.read(len(_TSFILE_MAGIC))
        except OSError as e:
            raise ValueError(f"Cannot open file {file!r}: {e}") from e
        if header != _TSFILE_MAGIC:
            raise ValueError(f"File {file!r} is not a valid TsFile (bad magic header).")
        return TsFileReader(file)

    @staticmethod
    def _require_table_model(file: str, schemas) -> None:
        if not schemas:
            raise ValueError(
                f"File {file!r} is a tree-model TsFile, which is not supported. "
                "Only table-model TsFiles can be loaded."
            )

    def _should_reraise(self, file: str, exc: BaseException) -> bool:
        """Apply ``on_bad_files`` policy. Returns True iff the caller should re-raise."""
        mode = self.config.on_bad_files
        if mode == "error":
            logger.error(f"Failed to read file '{file}' with error {type(exc).__name__}: {exc}")
            return True
        if mode == "warn":
            logger.warning(f"Skipping bad file '{file}'. {type(exc).__name__}: {exc}")
        else:
            logger.debug(f"Skipping bad file '{file}'. {type(exc).__name__}: {exc}")
        return False


class _SkipFile(Exception):
    """Internal sentinel raised to fast-path out of a per-file generator."""


class _MissingTableError(ValueError):
    """Raised when the requested table is absent from a particular file."""

    def __init__(self, table: Optional[str], available):
        super().__init__(f"Table {table!r} not found in file. Available tables: {available}")
