from datetime import date, datetime

import pyarrow as pa
import pytest

from datasets import load_dataset
from datasets.builder import InvalidConfigName
from datasets.data_files import DataFilesList
from datasets.packaged_modules.tsfile.tsfile import TsFileConfig


tsfile = pytest.importorskip("tsfile")


# ---------------------------------------------------------------------------
# Fixtures: construct small table-model TsFiles for the tests below.
# ---------------------------------------------------------------------------


def _write_single_device_tsfile(path: str) -> None:
    """One table, one device, two double fields."""
    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema = TableSchema(
        "mytable",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema)
        n = 5
        tablet = Tablet(
            ["device", "temperature", "humidity"],
            [TSDataType.STRING, TSDataType.DOUBLE, TSDataType.DOUBLE],
            n,
        )
        tablet.set_table_name("mytable")
        for i in range(n):
            tablet.add_timestamp(i, 1_700_000_000_000 + i * 1000)
            tablet.add_value_by_name("device", i, "d1")
            tablet.add_value_by_name("temperature", i, 20.0 + i)
            tablet.add_value_by_name("humidity", i, 50.0 + i)
        writer.write_table(tablet)
    finally:
        writer.close()


def _write_multi_device_tsfile(path: str) -> None:
    """One table, three devices, two fields each."""
    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema = TableSchema(
        "plant",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema)
        for device in ("d1", "d2", "d3"):
            tablet = Tablet(
                ["device", "temperature", "humidity"],
                [TSDataType.STRING, TSDataType.DOUBLE, TSDataType.DOUBLE],
                3,
            )
            tablet.set_table_name("plant")
            for i in range(3):
                tablet.add_timestamp(i, 1_700_000_000_000 + i * 1000)
                tablet.add_value_by_name("device", i, device)
                tablet.add_value_by_name("temperature", i, 10.0 + i)
                tablet.add_value_by_name("humidity", i, 50.0 + i)
            writer.write_table(tablet)
    finally:
        writer.close()


def _write_evolved_tsfile(path: str) -> None:
    """A second-day file: same table name as the single-device fixture but with
    an extra ``voltage`` field, exercising IoTDB's column-evolution behavior."""
    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema = TableSchema(
        "mytable",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("humidity", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("voltage", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema)
        n = 3
        tablet = Tablet(
            ["device", "temperature", "humidity", "voltage"],
            [TSDataType.STRING, TSDataType.DOUBLE, TSDataType.DOUBLE, TSDataType.DOUBLE],
            n,
        )
        tablet.set_table_name("mytable")
        for i in range(n):
            tablet.add_timestamp(i, 1_700_000_100_000 + i * 1000)
            tablet.add_value_by_name("device", i, "d1")
            tablet.add_value_by_name("temperature", i, 30.0 + i)
            tablet.add_value_by_name("humidity", i, 60.0 + i)
            tablet.add_value_by_name("voltage", i, 220.0 + i)
        writer.write_table(tablet)
    finally:
        writer.close()


def _write_int32_field_tsfile(path: str) -> None:
    """One table with an INT32 field — used to test type promotion."""
    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema = TableSchema(
        "mytable",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.INT32, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema)
        tablet = Tablet(
            ["device", "temperature"],
            [TSDataType.STRING, TSDataType.INT32],
            3,
        )
        tablet.set_table_name("mytable")
        for i in range(3):
            tablet.add_timestamp(i, 1_700_000_000_000 + i * 1000)
            tablet.add_value_by_name("device", i, "d1")
            tablet.add_value_by_name("temperature", i, 10 + i)
        writer.write_table(tablet)
    finally:
        writer.close()


def _write_int64_field_tsfile(path: str) -> None:
    """One table with an INT64 field — same column as INT32 fixture but widened."""
    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema = TableSchema(
        "mytable",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.INT64, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema)
        tablet = Tablet(
            ["device", "temperature"],
            [TSDataType.STRING, TSDataType.INT64],
            3,
        )
        tablet.set_table_name("mytable")
        for i in range(3):
            tablet.add_timestamp(i, 1_700_000_200_000 + i * 1000)
            tablet.add_value_by_name("device", i, "d1")
            tablet.add_value_by_name("temperature", i, 1_000_000 + i)
        writer.write_table(tablet)
    finally:
        writer.close()


def _write_two_tables_tsfile(path: str) -> None:
    """A file containing two distinct tables to exercise ``table_name``."""
    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema_a = TableSchema(
        "table_a",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("a", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )
    schema_b = TableSchema(
        "table_b",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("b", TSDataType.DOUBLE, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema_a)
        writer.register_table(schema_b)

        tablet_a = Tablet(["device", "a"], [TSDataType.STRING, TSDataType.DOUBLE], 2)
        tablet_a.set_table_name("table_a")
        for i in range(2):
            tablet_a.add_timestamp(i, 1_000 + i)
            tablet_a.add_value_by_name("device", i, "d1")
            tablet_a.add_value_by_name("a", i, float(i))
        writer.write_table(tablet_a)

        tablet_b = Tablet(["device", "b"], [TSDataType.STRING, TSDataType.DOUBLE], 2)
        tablet_b.set_table_name("table_b")
        for i in range(2):
            tablet_b.add_timestamp(i, 2_000 + i)
            tablet_b.add_value_by_name("device", i, "d1")
            tablet_b.add_value_by_name("b", i, 100.0 + i)
        writer.write_table(tablet_b)
    finally:
        writer.close()


def _write_all_types_tsfile(path: str) -> None:
    """One table with every supported TSDataType as columns."""
    from datetime import date as _date

    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema = TableSchema(
        "alltypes",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("tag", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("col_boolean", TSDataType.BOOLEAN, ColumnCategory.FIELD),
            ColumnSchema("col_int32", TSDataType.INT32, ColumnCategory.FIELD),
            ColumnSchema("col_int64", TSDataType.INT64, ColumnCategory.FIELD),
            ColumnSchema("col_float", TSDataType.FLOAT, ColumnCategory.FIELD),
            ColumnSchema("col_double", TSDataType.DOUBLE, ColumnCategory.FIELD),
            ColumnSchema("col_text", TSDataType.TEXT, ColumnCategory.FIELD),
            ColumnSchema("col_string", TSDataType.STRING, ColumnCategory.FIELD),
            ColumnSchema("col_timestamp", TSDataType.TIMESTAMP, ColumnCategory.FIELD),
            ColumnSchema("col_date", TSDataType.DATE, ColumnCategory.FIELD),
            ColumnSchema("col_blob", TSDataType.BLOB, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema)
        col_names = [
            "tag",
            "col_boolean",
            "col_int32",
            "col_int64",
            "col_float",
            "col_double",
            "col_text",
            "col_string",
            "col_timestamp",
            "col_date",
            "col_blob",
        ]
        col_types = [
            TSDataType.STRING,
            TSDataType.BOOLEAN,
            TSDataType.INT32,
            TSDataType.INT64,
            TSDataType.FLOAT,
            TSDataType.DOUBLE,
            TSDataType.TEXT,
            TSDataType.STRING,
            TSDataType.TIMESTAMP,
            TSDataType.DATE,
            TSDataType.BLOB,
        ]
        n = 3
        tablet = Tablet(col_names, col_types, n)
        tablet.set_table_name("alltypes")
        for i in range(n):
            tablet.add_timestamp(i, 1_700_000_000_000 + i * 1000)
            tablet.add_value_by_name("tag", i, "d1")
            tablet.add_value_by_name("col_boolean", i, i % 2 == 0)
            tablet.add_value_by_name("col_int32", i, 100 + i)
            tablet.add_value_by_name("col_int64", i, 1_000_000 + i)
            tablet.add_value_by_name("col_float", i, 1.5 + i)
            tablet.add_value_by_name("col_double", i, 100.5 + i)
            tablet.add_value_by_name("col_text", i, f"text_{i}")
            tablet.add_value_by_name("col_string", i, f"str_{i}")
            tablet.add_value_by_name("col_timestamp", i, 1_600_000_000_000 + i * 1000)
            tablet.add_value_by_name("col_date", i, _date(2024, 1, 1 + i))
            tablet.add_value_by_name("col_blob", i, f"blob{i}".encode())
        writer.write_table(tablet)
    finally:
        writer.close()


def _write_float_field_tsfile(path: str) -> None:
    """One table with a FLOAT field — used to test FLOAT→DOUBLE promotion."""
    from tsfile import (
        ColumnCategory,
        ColumnSchema,
        TableSchema,
        Tablet,
        TsFileWriter,
    )
    from tsfile.constants import TSDataType

    schema = TableSchema(
        "mytable",
        [
            ColumnSchema("time", TSDataType.TIMESTAMP, ColumnCategory.TIME),
            ColumnSchema("device", TSDataType.STRING, ColumnCategory.TAG),
            ColumnSchema("temperature", TSDataType.FLOAT, ColumnCategory.FIELD),
        ],
    )

    writer = TsFileWriter(path)
    try:
        writer.register_table(schema)
        tablet = Tablet(
            ["device", "temperature"],
            [TSDataType.STRING, TSDataType.FLOAT],
            3,
        )
        tablet.set_table_name("mytable")
        for i in range(3):
            tablet.add_timestamp(i, 1_700_000_300_000 + i * 1000)
            tablet.add_value_by_name("device", i, "d1")
            tablet.add_value_by_name("temperature", i, 1.5 + i)
        writer.write_table(tablet)
    finally:
        writer.close()


@pytest.fixture
def tsfile_path(tmp_path) -> str:
    path = tmp_path / "sample.tsfile"
    _write_single_device_tsfile(str(path))
    return str(path)


@pytest.fixture
def multi_device_tsfile_path(tmp_path) -> str:
    path = tmp_path / "multi.tsfile"
    _write_multi_device_tsfile(str(path))
    return str(path)


@pytest.fixture
def evolved_tsfile_path(tmp_path) -> str:
    path = tmp_path / "evolved.tsfile"
    _write_evolved_tsfile(str(path))
    return str(path)


@pytest.fixture
def two_tables_tsfile_path(tmp_path) -> str:
    path = tmp_path / "two_tables.tsfile"
    _write_two_tables_tsfile(str(path))
    return str(path)


@pytest.fixture
def all_types_tsfile_path(tmp_path) -> str:
    path = tmp_path / "alltypes.tsfile"
    _write_all_types_tsfile(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Config-level tests
# ---------------------------------------------------------------------------


def test_config_raises_when_invalid_name() -> None:
    with pytest.raises(InvalidConfigName, match="Bad characters"):
        _ = TsFileConfig(name="name-with-*-invalid-character")


@pytest.mark.parametrize("data_files", ["str_path", ["str_path"], DataFilesList(["str_path"], [()])])
def test_config_raises_when_invalid_data_files(data_files) -> None:
    with pytest.raises(ValueError, match="Expected a DataFilesDict"):
        _ = TsFileConfig(name="name", data_files=data_files)


def test_config_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        TsFileConfig(name="x", batch_size=0)


def test_config_rejects_empty_columns() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TsFileConfig(name="x", columns=[])


# ---------------------------------------------------------------------------
# End-to-end loading
# ---------------------------------------------------------------------------


def test_load_full_table(tsfile_path):
    dataset = load_dataset("tsfile", data_files=tsfile_path)["train"]

    # Native view: time + tag + field columns.
    assert dataset.column_names == ["time", "device", "temperature", "humidity"]
    assert len(dataset) == 5
    assert dataset["time"][0] == datetime(2023, 11, 14, 22, 13, 20)
    assert dataset["time"][-1] == datetime(2023, 11, 14, 22, 13, 24)
    assert dataset["device"] == ["d1"] * 5
    assert dataset["temperature"] == [20.0, 21.0, 22.0, 23.0, 24.0]


def test_load_with_columns_subset(tsfile_path):
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        columns=["temperature"],
    )["train"]

    # Time column is always preserved; user-requested order is honored.
    assert dataset.column_names == ["time", "temperature"]
    assert dataset["temperature"] == [20.0, 21.0, 22.0, 23.0, 24.0]


def test_load_with_time_range(tsfile_path):
    start = pa.scalar(datetime(2023, 11, 14, 22, 13, 21), type=pa.timestamp("ms"))
    end = pa.scalar(datetime(2023, 11, 14, 22, 13, 23), type=pa.timestamp("ms"))
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        start_time=start,
        end_time=end,
    )["train"]

    timestamps = dataset["time"]
    assert min(timestamps) >= datetime(2023, 11, 14, 22, 13, 21)
    assert max(timestamps) <= datetime(2023, 11, 14, 22, 13, 23)
    assert len(timestamps) == 3


def test_load_with_epoch_int_time_range(tsfile_path):
    """pa.scalar can be constructed from raw epoch integers."""
    start = pa.scalar(1_700_000_001_000, type=pa.timestamp("ms"))
    end = pa.scalar(1_700_000_003_000, type=pa.timestamp("ms"))
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        start_time=start,
        end_time=end,
    )["train"]

    timestamps = dataset["time"]
    assert min(timestamps) >= datetime(2023, 11, 14, 22, 13, 21)
    assert max(timestamps) <= datetime(2023, 11, 14, 22, 13, 23)
    assert len(timestamps) == 3


def test_load_with_string_timestamp_time_range(tsfile_path):
    """pa.scalar accepts string timestamps via datetime parsing."""
    import pandas as _pd

    start = pa.scalar(_pd.Timestamp("2023-11-14 22:13:21"), type=pa.timestamp("ms"))
    end = pa.scalar(_pd.Timestamp("2023-11-14 22:13:23"), type=pa.timestamp("ms"))
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        start_time=start,
        end_time=end,
    )["train"]

    timestamps = dataset["time"]
    assert min(timestamps) >= datetime(2023, 11, 14, 22, 13, 21)
    assert max(timestamps) <= datetime(2023, 11, 14, 22, 13, 23)
    assert len(timestamps) == 3


def test_load_multi_device_keeps_tag_column(multi_device_tsfile_path):
    dataset = load_dataset("tsfile", data_files=multi_device_tsfile_path)["train"]

    assert dataset.column_names == ["time", "device", "temperature", "humidity"]
    assert sorted(set(dataset["device"])) == ["d1", "d2", "d3"]
    assert len(dataset) == 9


def test_columns_are_lowercased(tsfile_path):
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        columns=["TEMPERATURE", "Humidity"],
    )["train"]
    assert dataset.column_names == ["time", "temperature", "humidity"]


def test_columns_subset_preserves_string_tag_type(tsfile_path):
    # Regression: when the user requests a STRING TAG column via `columns=[...]`,
    # the inferred Arrow type must match the file's actual dtype (string), not
    # silently fall back to float64 and crash inside ``pa.array``.
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        columns=["device", "temperature"],
    )["train"]

    assert dataset.column_names == ["time", "device", "temperature"]
    assert dataset.features["device"].dtype == "string"
    assert dataset.features["temperature"].dtype == "float64"
    assert dataset["device"] == ["d1"] * 5
    assert dataset["temperature"] == [20.0, 21.0, 22.0, 23.0, 24.0]


def test_load_all_supported_types(all_types_tsfile_path):
    """Verify that every TSDataType is correctly mapped through the full pipeline.

    Covers: BOOLEAN, INT32, INT64, FLOAT, DOUBLE, TEXT, STRING, TIMESTAMP, DATE, BLOB.
    """
    dataset = load_dataset("tsfile", data_files=all_types_tsfile_path)["train"]

    assert len(dataset) == 3
    expected_cols = [
        "time",
        "tag",
        "col_boolean",
        "col_int32",
        "col_int64",
        "col_float",
        "col_double",
        "col_text",
        "col_string",
        "col_timestamp",
        "col_date",
        "col_blob",
    ]
    assert dataset.column_names == expected_cols

    # STRING (tag)
    assert dataset["tag"] == ["d1", "d1", "d1"]
    # BOOLEAN
    assert dataset["col_boolean"] == [True, False, True]
    # INT32
    assert dataset["col_int32"] == [100, 101, 102]
    # INT64
    assert dataset["col_int64"] == [1_000_000, 1_000_001, 1_000_002]
    # FLOAT (values exact in float32)
    assert dataset["col_float"] == [1.5, 2.5, 3.5]
    # DOUBLE
    assert dataset["col_double"] == [100.5, 101.5, 102.5]
    # TEXT
    assert dataset["col_text"] == ["text_0", "text_1", "text_2"]
    # STRING (as field)
    assert dataset["col_string"] == ["str_0", "str_1", "str_2"]
    # TIMESTAMP (as field) — epoch millis → datetime
    assert dataset["col_timestamp"][0] == datetime(2020, 9, 13, 12, 26, 40)
    # DATE
    assert dataset["col_date"][0] == date(2024, 1, 1)
    assert dataset["col_date"][2] == date(2024, 1, 3)
    # BLOB
    assert dataset["col_blob"][0] == b"blob0"
    assert dataset["col_blob"][2] == b"blob2"


# ---------------------------------------------------------------------------
# Multi-file behavior: schema evolution & null-fill for absent columns
# ---------------------------------------------------------------------------


def test_load_schema_evolution_unions_columns(tsfile_path, evolved_tsfile_path):
    dataset = load_dataset(
        "tsfile",
        data_files=[tsfile_path, evolved_tsfile_path],
    )["train"]

    # Old file lacked `voltage` -> those rows must be null; new column appears in schema.
    assert "voltage" in dataset.column_names
    assert len(dataset) == 5 + 3

    voltages = dataset["voltage"]
    # First 5 rows came from the old file -> null; remaining 3 from the new file.
    assert voltages[:5] == [None] * 5
    assert voltages[5:] == [220.0, 221.0, 222.0]


def test_columns_request_unknown_field_filled_with_null(tsfile_path):
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        columns=["temperature", "voltage"],  # voltage absent from this file
    )["train"]

    assert dataset.column_names == ["time", "temperature", "voltage"]
    assert dataset["voltage"] == [None] * 5
    assert dataset["temperature"] == [20.0, 21.0, 22.0, 23.0, 24.0]


def test_columns_all_absent_still_returns_time_and_nulls(tsfile_path):
    """When every requested column is absent from the file, the time column
    still comes through and the missing columns are filled with nulls.
    This verifies that:
    1. ``to_dataframe(column_names=[])`` is called (no matching columns in this file),
    2. ``_dataframe_to_arrow`` correctly null-fills all requested columns,
    3. The time column retains its real values from the file.
    """
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        columns=["nonexistent_a", "nonexistent_b"],
    )["train"]

    assert dataset.column_names == ["time", "nonexistent_a", "nonexistent_b"]
    assert len(dataset) == 5
    # Time still has real values from the file.
    assert dataset["time"][0] == datetime(2023, 11, 14, 22, 13, 20)
    # All requested columns are null because they don't exist in the file.
    assert dataset["nonexistent_a"] == [None] * 5
    assert dataset["nonexistent_b"] == [None] * 5


def test_type_promotion_int32_to_int64(tmp_path):
    """When two files have the same column with INT32 and INT64, the merged schema
    should use INT64 (the wider type), matching IoTDB's ALTER COLUMN behavior."""
    int32_path = str(tmp_path / "narrow.tsfile")
    int64_path = str(tmp_path / "wide.tsfile")
    _write_int32_field_tsfile(int32_path)
    _write_int64_field_tsfile(int64_path)

    dataset = load_dataset("tsfile", data_files=[int32_path, int64_path])["train"]

    assert dataset.features["temperature"].dtype == "int64"
    assert len(dataset) == 6
    # INT32 file rows (10, 11, 12) safely widened to int64.
    assert dataset["temperature"][:3] == [10, 11, 12]
    # INT64 file rows.
    assert dataset["temperature"][3:] == [1_000_000, 1_000_001, 1_000_002]


def test_type_promotion_float_to_double(tmp_path):
    """FLOAT→DOUBLE promotion when two files have the same column."""
    float_path = str(tmp_path / "narrow.tsfile")
    double_path = str(tmp_path / "wide.tsfile")
    _write_float_field_tsfile(float_path)
    _write_single_device_tsfile(double_path)

    dataset = load_dataset("tsfile", data_files=[float_path, double_path])["train"]

    assert dataset.features["temperature"].dtype == "float64"
    assert len(dataset) == 3 + 5
    # FLOAT rows safely widened to double.
    assert dataset["temperature"][:3] == [1.5, 2.5, 3.5]
    assert dataset["temperature"][3:] == [20.0, 21.0, 22.0, 23.0, 24.0]


def test_type_promotion_int32_to_double(tmp_path):
    """INT32→DOUBLE promotion (two-step widening across the rank table)."""
    int32_path = str(tmp_path / "int.tsfile")
    double_path = str(tmp_path / "double.tsfile")
    _write_int32_field_tsfile(int32_path)
    _write_single_device_tsfile(double_path)

    dataset = load_dataset("tsfile", data_files=[int32_path, double_path])["train"]

    assert dataset.features["temperature"].dtype == "float64"
    assert len(dataset) == 3 + 5
    # INT32 rows cast to double.
    assert dataset["temperature"][:3] == [10.0, 11.0, 12.0]
    assert dataset["temperature"][3:] == [20.0, 21.0, 22.0, 23.0, 24.0]


# ---------------------------------------------------------------------------
# Multi-table file: explicit table_name selection
# ---------------------------------------------------------------------------


def test_default_table_is_first(two_tables_tsfile_path):
    dataset = load_dataset("tsfile", data_files=two_tables_tsfile_path)["train"]
    # `table_a` is registered first -> default pick.
    assert "a" in dataset.column_names
    assert "b" not in dataset.column_names


def test_explicit_table_name(two_tables_tsfile_path):
    dataset = load_dataset(
        "tsfile",
        data_files=two_tables_tsfile_path,
        table_name="table_b",
    )["train"]
    assert "b" in dataset.column_names
    assert "a" not in dataset.column_names


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_on_bad_files_skip(tmp_path, tsfile_path):
    bad_path = tmp_path / "broken.tsfile"
    bad_path.write_bytes(b"not a real tsfile")

    dataset = load_dataset(
        "tsfile",
        data_files=[tsfile_path, str(bad_path)],
        on_bad_files="skip",
    )["train"]
    assert len(dataset) == 5


def test_on_bad_files_default_raises(tmp_path, tsfile_path):
    bad_path = tmp_path / "broken.tsfile"
    bad_path.write_bytes(b"not a real tsfile")

    with pytest.raises(Exception):
        load_dataset("tsfile", data_files=[tsfile_path, str(bad_path)])
