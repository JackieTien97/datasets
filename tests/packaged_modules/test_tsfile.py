import pytest

tsfile_lib = pytest.importorskip("tsfile", reason="test requires tsfile")

import numpy as np
import pandas as pd
import pyarrow as pa
from tsfile import (
    ColumnCategory,
    ColumnSchema,
    TableSchema,
    TSDataType,
    TsFileTableWriter,
)

from datasets import Value, load_dataset
from datasets.builder import InvalidConfigName
from datasets.data_files import DataFilesList
from datasets.packaged_modules.tsfile.tsfile import TsFile, TsFileConfig


def _write_single_table_file(path):
    schema = TableSchema(
        "sensor",
        [
            ColumnSchema("region", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    df = pd.DataFrame(
        {
            "time": [1000, 2000, 3000, 4000, 5000],
            "region": ["asia", "asia", "asia", "asia", "asia"],
            "temperature": [25.0, 26.0, 27.0, 28.0, 29.0],
            "humidity": [60.0, 58.0, 55.0, 53.0, 50.0],
        }
    )
    with TsFileTableWriter(str(path), schema) as writer:
        writer.write_dataframe(df)
    return str(path)


def _write_multi_device_file(path):
    schema = TableSchema(
        "sensor",
        [
            ColumnSchema("region", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    df = pd.DataFrame(
        {
            "time": [1000, 2000, 3000, 1000, 2000, 3000],
            "region": ["asia", "asia", "asia", "eu", "eu", "eu"],
            "temperature": [25.0, 26.0, 27.0, 18.0, 17.0, 16.0],
            "humidity": [60.0, 58.0, 55.0, 75.0, 72.0, 70.0],
        }
    )
    with TsFileTableWriter(str(path), schema) as writer:
        writer.write_dataframe(df)
    return str(path)


@pytest.fixture
def tsfile_single_table(tmp_path):
    return _write_single_table_file(tmp_path / "single.tsfile")


@pytest.fixture
def tsfile_multi_device(tmp_path):
    return _write_multi_device_file(tmp_path / "multi_device.tsfile")


@pytest.fixture
def tsfile_multiple_files(tmp_path):
    schema = TableSchema(
        "sensor",
        [
            ColumnSchema("region", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    paths = []
    for i in range(2):
        path = tmp_path / f"part{i}.tsfile"
        df = pd.DataFrame(
            {
                "time": [i * 1000 + 1, i * 1000 + 2, i * 1000 + 3],
                "region": ["asia", "asia", "asia"],
                "temperature": [20.0 + i, 21.0 + i, 22.0 + i],
            }
        )
        with TsFileTableWriter(str(path), schema) as writer:
            writer.write_dataframe(df)
        paths.append(str(path))
    return paths


def test_config_raises_when_invalid_name():
    with pytest.raises(InvalidConfigName, match="Bad characters"):
        TsFileConfig(name="name-with-*-invalid-character")


@pytest.mark.parametrize("data_files", ["str_path", ["str_path"], DataFilesList(["str_path"], [()])])
def test_config_raises_when_invalid_data_files(data_files):
    with pytest.raises(ValueError, match="Expected a DataFilesDict"):
        TsFileConfig(name="name", data_files=data_files)


def test_tsfile_generate_tables(tsfile_single_table):
    builder = TsFile(table_name="sensor")
    generator = builder._generate_tables([tsfile_single_table])
    tables = [table for _, table in generator]
    pa_table = pa.concat_tables(tables)

    assert "time" in pa_table.column_names
    assert "region" in pa_table.column_names
    assert "temperature" in pa_table.column_names
    assert "humidity" in pa_table.column_names
    assert pa_table.num_rows == 5
    assert pa_table.column("time").to_pylist() == [1000, 2000, 3000, 4000, 5000]
    assert pa_table.column("temperature").to_pylist() == [25.0, 26.0, 27.0, 28.0, 29.0]
    assert pa_table.column("humidity").to_pylist() == [60.0, 58.0, 55.0, 53.0, 50.0]
    assert pa_table.column("region").to_pylist() == ["asia"] * 5


def test_tsfile_features_inference(tsfile_single_table):
    from datasets.packaged_modules.tsfile.tsfile import _infer_features_from_tsfile

    features, table_name = _infer_features_from_tsfile(tsfile_single_table, None, None)
    assert table_name == "sensor"
    assert "time" in features
    assert "region" in features
    assert "temperature" in features
    assert "humidity" in features
    assert features["time"] == Value("int64")
    assert features["region"] == Value("string")
    assert features["temperature"] == Value("float64")
    assert features["humidity"] == Value("float64")


def test_tsfile_multi_device(tsfile_multi_device):
    builder = TsFile(table_name="sensor")
    generator = builder._generate_tables([tsfile_multi_device])
    tables = [table for _, table in generator]
    pa_table = pa.concat_tables(tables)

    assert pa_table.num_rows == 6
    data = pa_table.to_pydict()
    regions = data["region"]
    assert "asia" in regions
    assert "eu" in regions
    asia_temps = [data["temperature"][i] for i in range(6) if data["region"][i] == "asia"]
    assert asia_temps == [25.0, 26.0, 27.0]
    eu_temps = [data["temperature"][i] for i in range(6) if data["region"][i] == "eu"]
    assert eu_temps == [18.0, 17.0, 16.0]


def test_tsfile_select_columns(tsfile_single_table):
    builder = TsFile(table_name="sensor", columns=["temperature"])
    generator = builder._generate_tables([tsfile_single_table])
    tables = [table for _, table in generator]
    pa_table = pa.concat_tables(tables)

    assert "time" in pa_table.column_names
    assert "region" in pa_table.column_names
    assert "temperature" in pa_table.column_names
    assert "humidity" not in pa_table.column_names
    assert pa_table.num_rows == 5


def test_resolve_table_name_single_table():
    from datasets.packaged_modules.tsfile.tsfile import _resolve_table_name
    from types import SimpleNamespace

    df = SimpleNamespace(_index=SimpleNamespace(table_entries={"weather": object()}))
    assert _resolve_table_name(df, None) == "weather"
    assert _resolve_table_name(df, "weather") == "weather"


def test_resolve_table_name_multi_table_requires_explicit():
    from datasets.packaged_modules.tsfile.tsfile import _resolve_table_name
    from types import SimpleNamespace

    df = SimpleNamespace(_index=SimpleNamespace(table_entries={"weather": object(), "traffic": object()}))
    with pytest.raises(ValueError, match="multiple tables"):
        _resolve_table_name(df, None)
    assert _resolve_table_name(df, "traffic") == "traffic"


def test_resolve_table_name_invalid_name():
    from datasets.packaged_modules.tsfile.tsfile import _resolve_table_name
    from types import SimpleNamespace

    df = SimpleNamespace(_index=SimpleNamespace(table_entries={"weather": object()}))
    with pytest.raises(ValueError, match="not found"):
        _resolve_table_name(df, "nonexistent")


def test_tsfile_load_dataset(tsfile_single_table):
    ds = load_dataset("tsfile", data_files=tsfile_single_table, table_name="sensor", split="train")
    assert len(ds) == 5
    assert "time" in ds.column_names
    assert "region" in ds.column_names
    assert "temperature" in ds.column_names
    assert "humidity" in ds.column_names
    assert ds[0]["time"] == 1000
    assert ds[0]["temperature"] == 25.0
    assert ds[0]["region"] == "asia"


def test_tsfile_multiple_files(tsfile_multiple_files):
    ds = load_dataset("tsfile", data_files=tsfile_multiple_files, table_name="sensor", split="train")
    assert len(ds) == 6
    assert "time" in ds.column_names
    assert "temperature" in ds.column_names


def test_tsfile_batch_size(tsfile_single_table):
    builder = TsFile(table_name="sensor", batch_size=2)
    generator = builder._generate_tables([tsfile_single_table])
    tables = list(generator)
    assert len(tables) >= 3
    for _, table in tables[:-1]:
        assert table.num_rows <= 2


def test_tsfile_batch_size_exceeds_data_length(tsfile_single_table):
    builder = TsFile(table_name="sensor", batch_size=1000)
    generator = builder._generate_tables([tsfile_single_table])
    tables = list(generator)
    assert len(tables) == 1
    _, table = tables[0]
    assert table.num_rows == 5


def test_tsfile_no_tag_columns(tmp_path):
    schema = TableSchema(
        "metrics",
        [
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    df = pd.DataFrame(
        {
            "time": [1000, 2000, 3000],
            "temperature": [25.0, 26.0, 27.0],
            "humidity": [60.0, 58.0, 55.0],
        }
    )
    path = str(tmp_path / "no_tags.tsfile")
    with TsFileTableWriter(path, schema) as writer:
        writer.write_dataframe(df)

    builder = TsFile(table_name="metrics")
    generator = builder._generate_tables([path])
    tables = [table for _, table in generator]
    pa_table = pa.concat_tables(tables)

    assert "time" in pa_table.column_names
    assert "temperature" in pa_table.column_names
    assert "humidity" in pa_table.column_names
    assert pa_table.num_rows == 3
    assert pa_table.column("temperature").to_pylist() == [25.0, 26.0, 27.0]


def test_tsfile_mixed_field_types(tmp_path):
    schema = TableSchema(
        "device",
        [
            ColumnSchema("name", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("status", TSDataType.STRING, ColumnCategory.FIELD),
            ColumnSchema("label", TSDataType.TEXT, ColumnCategory.FIELD),
            ColumnSchema("active", TSDataType.BOOLEAN, ColumnCategory.FIELD),
            ColumnSchema("count", TSDataType.INT32, ColumnCategory.FIELD),
            ColumnSchema("total", TSDataType.INT64, ColumnCategory.FIELD),
            ColumnSchema("score", TSDataType.FLOAT, ColumnCategory.FIELD),
        ],
    )
    df = pd.DataFrame(
        {
            "time": [1000, 2000],
            "name": ["dev1", "dev1"],
            "status": ["ok", "warn"],
            "label": ["alpha", "beta"],
            "active": [True, False],
            "count": np.array([10, 20], dtype=np.int32),
            "total": np.array([100, 200], dtype=np.int64),
            "score": np.array([1.5, 2.5], dtype=np.float32),
        }
    )
    path = str(tmp_path / "mixed_types.tsfile")
    with TsFileTableWriter(path, schema) as writer:
        writer.write_dataframe(df)

    from datasets.packaged_modules.tsfile.tsfile import _infer_features_from_tsfile

    features, table_name = _infer_features_from_tsfile(path, None, None)
    assert table_name == "device"
    assert features["status"] == Value("string")
    assert features["label"] == Value("string")
    assert features["active"] == Value("bool")
    assert features["count"] == Value("int32")
    assert features["total"] == Value("int64")
    assert features["score"] == Value("float32")

    builder = TsFile(table_name="device")
    generator = builder._generate_tables([path])
    tables = [table for _, table in generator]
    pa_table = pa.concat_tables(tables)

    assert pa_table.num_rows == 2
    assert pa_table.column("status").to_pylist() == ["ok", "warn"]
    assert pa_table.column("label").to_pylist() == ["alpha", "beta"]
    assert pa_table.column("active").to_pylist() == [True, False]
    assert pa_table.column("count").to_pylist() == [10, 20]
