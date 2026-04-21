from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa

import datasets
from datasets.builder import Key
from datasets.table import table_cast


logger = datasets.utils.logging.get_logger(__name__)


@dataclass
class TsFileConfig(datasets.BuilderConfig):
    """BuilderConfig for TsFile.

    Args:
        features (`Features`, *optional*):
            Cast the data to `features`.
        columns (`list[str]`, *optional*):
            List of field columns to load. Tag columns and the time column are always included.
            All field columns are loaded by default.
        table_name (`str`, *optional*):
            Name of the table to load from the TsFile. Required when the TsFile contains
            multiple tables. When the TsFile contains a single table, it is selected automatically.
        batch_size (`int`, *optional*):
            Maximum number of rows per yielded Arrow table. When ``None``, each device's
            data is yielded as a single table.
    """

    features: Optional[datasets.Features] = None
    columns: Optional[list[str]] = None
    table_name: Optional[str] = None
    batch_size: Optional[int] = None


def _resolve_table_name(df, config_table_name: Optional[str]) -> str:
    table_names = list(df._index.table_entries.keys())
    if not table_names:
        raise ValueError("No tables found in TsFile")
    if config_table_name is not None:
        if config_table_name not in table_names:
            raise ValueError(
                f"Table '{config_table_name}' not found in TsFile. Available tables: {table_names}"
            )
        return config_table_name
    if len(table_names) == 1:
        return table_names[0]
    raise ValueError(
        f"TsFile contains multiple tables: {table_names}. "
        f"Please specify which table to load using the 'table_name' parameter."
    )


def _tsfile_dtype_to_datasets_value(data_type) -> datasets.Value:
    from tsfile.constants import TSDataType

    mapping = {
        TSDataType.BOOLEAN: "bool",
        TSDataType.INT32: "int32",
        TSDataType.INT64: "int64",
        TSDataType.FLOAT: "float32",
        TSDataType.DOUBLE: "float64",
        TSDataType.TIMESTAMP: "int64",
        TSDataType.TEXT: "string",
        TSDataType.STRING: "string",
        TSDataType.DATE: "string",
        TSDataType.BLOB: "binary",
    }
    dtype_str = mapping.get(data_type)
    if dtype_str is None:
        logger.warning(f"Unknown TsFile data type '{data_type}', defaulting to float64")
        dtype_str = "float64"
    return datasets.Value(dtype_str)


def _infer_features_from_tsfile(file_path: str, config_table_name: Optional[str], config_columns: Optional[list[str]]) -> tuple:
    from tsfile import TsFileDataFrame
    from tsfile.constants import ColumnCategory

    df = TsFileDataFrame(file_path, show_progress=False)
    try:
        target_table = _resolve_table_name(df, config_table_name)
        table_entry = df._index.table_entries[target_table]

        feature_dict = {"time": datasets.Value("int64")}
        for tag_col in table_entry.tag_columns:
            feature_dict[tag_col] = datasets.Value("string")

        reader = next(iter(df._readers.values()))
        table_schemas = reader._reader.get_all_table_schemas()
        table_schema = table_schemas[target_table]
        field_type_map = {}
        for column_schema in table_schema.get_columns():
            cat = column_schema.get_category()
            if cat == ColumnCategory.FIELD:
                field_type_map[column_schema.get_column_name()] = column_schema.get_data_type()

        for field_col in table_entry.field_columns:
            if config_columns is not None and field_col not in config_columns:
                continue
            data_type = field_type_map.get(field_col)
            feature_dict[field_col] = _tsfile_dtype_to_datasets_value(data_type) if data_type else datasets.Value("float64")

        return datasets.Features(feature_dict), target_table
    finally:
        df.close()


def _group_series_by_device(df, target_table: str, config_columns: Optional[list[str]]) -> list:
    devices = defaultdict(list)
    for series_ref in df._index.series_refs_ordered:
        device_idx, field_idx = series_ref
        device_key = df._index.device_order[device_idx]
        table_name, tag_values = device_key
        if table_name != target_table:
            continue
        table_entry = df._index.table_entries[table_name]
        field_name = table_entry.field_columns[field_idx]
        if config_columns is not None and field_name not in config_columns:
            continue
        series_name = df._build_series_name(series_ref)
        devices[(table_name, tag_values)].append((series_name, field_name, series_ref))
    return list(devices.items())


class TsFile(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = TsFileConfig

    def _info(self):
        if (
            self.config.columns is not None
            and self.config.features is not None
            and not set(self.config.columns).issubset(set(self.config.features))
        ):
            raise ValueError(
                f"The columns parameter contains columns not present in features: "
                f"{set(self.config.columns) - set(self.config.features)}"
            )
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if self.info.features is None:
                for file in files:
                    try:
                        features, table_name = _infer_features_from_tsfile(
                            file, self.config.table_name, self.config.columns
                        )
                        self.info.features = features
                        if self.config.table_name is None:
                            self.config.table_name = table_name
                        break
                    except Exception as e:
                        logger.error(f"Failed to read schema from '{file}': {e}")
                        raise
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_shards(self, files):
        yield from files

    def _generate_tables(self, files):
        from tsfile import TsFileDataFrame

        for file_idx, file in enumerate(files):
            batch_counter = 0
            try:
                df = TsFileDataFrame(file, show_progress=False)
                try:
                    target_table = _resolve_table_name(df, self.config.table_name)
                    table_entry = df._index.table_entries[target_table]
                    device_groups = _group_series_by_device(df, target_table, self.config.columns)

                    for (table_name, tag_values), series_list in device_groups:
                        if not series_list:
                            continue

                        series_names = [s[0] for s in series_list]
                        field_names = [s[1] for s in series_list]

                        aligned = df.loc[:, series_names]
                        if len(aligned) == 0:
                            continue

                        data = {"time": aligned.timestamps}
                        for i, tag_col in enumerate(table_entry.tag_columns):
                            tag_val = tag_values[i] if i < len(tag_values) else None
                            data[tag_col] = [str(tag_val) if tag_val is not None else None] * len(aligned.timestamps)
                        for col_idx, field_name in enumerate(field_names):
                            data[field_name] = aligned.values[:, col_idx]

                        full_table = pa.table(data)
                        batch_size = self.config.batch_size or len(full_table)

                        for start in range(0, len(full_table), batch_size):
                            end = min(start + batch_size, len(full_table))
                            pa_table = full_table.slice(start, end - start)
                            yield Key(file_idx, batch_counter), self._cast_table(pa_table)
                            batch_counter += 1
                finally:
                    df.close()
            except Exception as e:
                logger.error(f"Failed to read file '{file}' with error {type(e).__name__}: {e}")
                raise
