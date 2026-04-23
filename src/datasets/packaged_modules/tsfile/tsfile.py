from dataclasses import dataclass
from typing import Iterator, Literal, Optional

import pandas as pd
import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


def _tsdatatype_to_arrow(ts_dtype) -> pa.DataType:
    """Map a tsfile ``TSDataType`` to its closest Arrow type."""
    from tsfile.constants import TSDataType

    mapping = {
        TSDataType.BOOLEAN: pa.bool_(),
        TSDataType.INT32: pa.int32(),
        TSDataType.INT64: pa.int64(),
        TSDataType.TIMESTAMP: pa.int64(),
        TSDataType.FLOAT: pa.float32(),
        TSDataType.DOUBLE: pa.float64(),
        TSDataType.TEXT: pa.string(),
        TSDataType.STRING: pa.string(),
        TSDataType.DATE: pa.string(),
        TSDataType.BLOB: pa.binary(),
    }
    return mapping.get(ts_dtype, pa.string())


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
        start_time (`int`, *optional*):
            Inclusive lower bound for the timestamp range. Defaults to no lower
            bound.
        end_time (`int`, *optional*):
            Inclusive upper bound for the timestamp range. Defaults to no upper
            bound.
        batch_size (`int`, *optional*, defaults to 100_000):
            Maximum number of rows per Arrow record batch. Larger values reduce
            per-batch overhead at the cost of more memory.
        features (`Features`, *optional*):
            Final Features schema. When provided, schema inference (and the
            associated metadata scan over input files) is skipped entirely.
        on_bad_files (`Literal["error", "warn", "skip"]`, *optional*, defaults to "error"):
            How to react when a file cannot be opened or does not contain the
            requested table.
    """

    table_name: Optional[str] = None
    columns: Optional[list[str]] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    batch_size: int = 100_000
    features: Optional[datasets.Features] = None
    on_bad_files: Literal["error", "warn", "skip"] = "error"

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be a positive integer, got {self.batch_size}")
        if self.columns is not None and len(self.columns) == 0:
            raise ValueError("`columns` must be a non-empty list when provided.")


class TsFile(datasets.ArrowBasedBuilder):
    """Apache TsFile builder (table model).

    Each input file is read through ``tsfile.to_dataframe`` in streaming mode and
    emitted as Arrow record batches that follow the union of the table schema
    seen across all files (IoTDB schema evolution: columns absent from a given
    file are filled with nulls).
    """

    BUILDER_CONFIG_CLASS = TsFileConfig

    def _info(self):
        if (
            self.config.columns is not None
            and self.config.features is not None
            and set(self.config.columns) != set(self.config.features)
        ):
            raise ValueError(
                "The columns and features argument must contain the same columns, but got ",
                f"{self.config.columns} and {self.config.features}",
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
        for split_name, files in data_files.items():
            files = list(files)
            if self.info.features is None:
                resolved_table, features = self._infer_features(files)
                if features is None:
                    raise ValueError(
                        "Could not infer schema from any of the provided files. "
                        "Set `features` explicitly or check the input files."
                    )
                self._resolved_table_name = resolved_table
                self.info.features = features
            elif self._resolved_table_name is None:
                # User pinned `features` but not `table_name`: still need a table to query.
                self._resolved_table_name = self._discover_first_table(files)

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

        Strategy:
        - ``columns`` provided (with or without ``table_name``) -> no file
          scanning needed for column collection.  Missing columns are filled
          with nulls (``float64``) at read time.  If ``table_name`` is also
          absent, a single file is opened only to discover the table name.
        - ``columns`` not provided -> scan every file and union the table
          columns so that schema-evolved files are supported.
        """
        from tsfile.constants import TIME_COLUMN, ColumnCategory

        wanted_table = self._resolved_table_name
        requested_columns = self._requested_columns

        # User specified columns → no need to collect column schemas from files.
        # Missing columns will be filled with nulls at read time.
        if requested_columns is not None:
            resolved_table = wanted_table
            if resolved_table is None:
                resolved_table = self._discover_first_table(files)
            if resolved_table is None:
                return None, None
            return resolved_table, self._build_features(TIME_COLUMN, {}, requested_columns)

        # User did not specify columns → full scan to union all columns.
        merged_columns: dict = {}  # name -> tsfile.ColumnSchema (lazy import)
        time_column_name: Optional[str] = None
        resolved_table: Optional[str] = wanted_table

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
                        merged_columns.setdefault(name, col)
            except Exception as e:
                if self._should_reraise(file, e):
                    raise
                continue

        if resolved_table is None:
            return None, None

        time_field_name = time_column_name or TIME_COLUMN
        return resolved_table, self._build_features(time_field_name, merged_columns, None)

    def _build_features(
        self,
        time_field_name: str,
        merged_columns: dict,
        requested_columns: Optional[list[str]],
    ) -> datasets.Features:
        """Assemble the Arrow schema and wrap it in ``Features``."""
        from tsfile.constants import ColumnCategory

        fields: list[pa.Field] = [pa.field(time_field_name, pa.int64())]

        if requested_columns is not None:
            # User specified columns — all default to float64.
            for name in requested_columns:
                if name != time_field_name:
                    fields.append(pa.field(name, pa.float64()))
        else:
            # No columns specified — use merged schema.
            # Ordering: tags first, then fields, each in insertion order.
            tag_cols = [c for c in merged_columns.values() if c.get_category() == ColumnCategory.TAG]
            field_cols = [c for c in merged_columns.values() if c.get_category() == ColumnCategory.FIELD]
            for col in (*tag_cols, *field_cols):
                fields.append(pa.field(col.get_column_name(), _tsdatatype_to_arrow(col.get_data_type())))

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
            columns_to_read = requested_present or None
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
        df = df.rename(columns=str.lower)
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
