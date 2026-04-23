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
    assert dataset["time"][0] == 1_700_000_000_000
    assert dataset["time"][-1] == 1_700_000_000_000 + 4 * 1000
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
    start = 1_700_000_000_000 + 1000
    end = 1_700_000_000_000 + 3000
    dataset = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        start_time=start,
        end_time=end,
    )["train"]

    timestamps = dataset["time"]
    assert min(timestamps) >= start
    assert max(timestamps) <= end
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
