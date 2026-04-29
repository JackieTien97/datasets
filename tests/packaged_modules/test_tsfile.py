import pytest

from datasets import load_dataset
from datasets.builder import InvalidConfigName
from datasets.data_files import DataFilesList
from datasets.packaged_modules.tsfile.tsfile import TsFileConfig


tsfile = pytest.importorskip("tsfile")


def _write_sample_tsfile(path: str) -> None:
    """Create a small TsFile with one table, one tag value, two numeric fields."""
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
    """Create a TsFile with three devices and two fields each."""
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


@pytest.fixture
def tsfile_path(tmp_path) -> str:
    path = tmp_path / "sample.tsfile"
    _write_sample_tsfile(str(path))
    return str(path)


@pytest.fixture
def multi_device_tsfile_path(tmp_path) -> str:
    path = tmp_path / "multi.tsfile"
    _write_multi_device_tsfile(str(path))
    return str(path)


@pytest.fixture
def tsfile_series_names(tsfile_path) -> list[str]:
    from tsfile import TsFileDataFrame

    df = TsFileDataFrame(tsfile_path, show_progress=False)
    try:
        return df.list_timeseries()
    finally:
        df.close()


def test_config_raises_when_invalid_name() -> None:
    with pytest.raises(InvalidConfigName, match="Bad characters"):
        _ = TsFileConfig(name="name-with-*-invalid-character")


@pytest.mark.parametrize("data_files", ["str_path", ["str_path"], DataFilesList(["str_path"], [()])])
def test_config_raises_when_invalid_data_files(data_files) -> None:
    with pytest.raises(ValueError, match="Expected a DataFilesDict"):
        _ = TsFileConfig(name="name", data_files=data_files)


def test_load_tsfile_dataset(tsfile_path, tsfile_series_names):
    dataset_dict = load_dataset("tsfile", data_files=tsfile_path)
    assert "train" in dataset_dict
    dataset = dataset_dict["train"]

    assert "timestamp" in dataset.column_names
    for name in tsfile_series_names:
        assert name in dataset.column_names

    assert len(dataset) == 5
    assert dataset["timestamp"][0] == 1_700_000_000_000
    assert dataset["timestamp"][-1] == 1_700_000_000_000 + 4 * 1000


def test_load_tsfile_dataset_with_columns(tsfile_path, tsfile_series_names):
    selected = tsfile_series_names[:1]
    dataset_dict = load_dataset("tsfile", data_files=tsfile_path, columns=selected)
    dataset = dataset_dict["train"]

    assert dataset.column_names == ["timestamp", *selected]
    for name in tsfile_series_names[1:]:
        assert name not in dataset.column_names


def test_load_tsfile_dataset_with_time_range(tsfile_path):
    start = 1_700_000_000_000 + 1000
    end = 1_700_000_000_000 + 3000
    dataset_dict = load_dataset(
        "tsfile",
        data_files=tsfile_path,
        start_time=start,
        end_time=end,
    )
    dataset = dataset_dict["train"]

    timestamps = dataset["timestamp"]
    assert min(timestamps) >= start
    assert max(timestamps) <= end
    assert len(timestamps) == 3


def test_load_tsfile_dataset_on_bad_files_skip(tmp_path, tsfile_path):
    bad_path = tmp_path / "broken.tsfile"
    bad_path.write_bytes(b"not a real tsfile")

    dataset_dict = load_dataset(
        "tsfile",
        data_files=[tsfile_path, str(bad_path)],
        on_bad_files="skip",
    )
    dataset = dataset_dict["train"]
    assert len(dataset) == 5


def test_load_tsfile_dataset_with_devices_filter(multi_device_tsfile_path):
    dataset = load_dataset("tsfile", data_files=multi_device_tsfile_path, devices=["d1"])["train"]
    assert dataset.column_names == ["timestamp", "plant.d1.temperature", "plant.d1.humidity"]


def test_load_tsfile_dataset_with_fields_filter(multi_device_tsfile_path):
    dataset = load_dataset("tsfile", data_files=multi_device_tsfile_path, fields=["temperature"])["train"]
    assert dataset.column_names == [
        "timestamp",
        "plant.d1.temperature",
        "plant.d2.temperature",
        "plant.d3.temperature",
    ]


def test_load_tsfile_dataset_with_devices_and_fields_filter(multi_device_tsfile_path):
    dataset = load_dataset(
        "tsfile",
        data_files=multi_device_tsfile_path,
        devices=["d2", "d3"],
        fields=["humidity"],
    )["train"]
    assert dataset.column_names == ["timestamp", "plant.d2.humidity", "plant.d3.humidity"]


def test_load_tsfile_dataset_with_path_prefix(multi_device_tsfile_path):
    dataset = load_dataset("tsfile", data_files=multi_device_tsfile_path, path_prefix="plant.d1")["train"]
    assert dataset.column_names == ["timestamp", "plant.d1.temperature", "plant.d1.humidity"]


def test_load_tsfile_dataset_filters_no_match_raises(multi_device_tsfile_path):
    with pytest.raises(ValueError, match="No timeseries matched"):
        load_dataset("tsfile", data_files=multi_device_tsfile_path, devices=["does_not_exist"])


def test_config_columns_conflicts_with_filters() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        TsFileConfig(name="x", columns=["a"], devices=["d1"])
