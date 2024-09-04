from drum_transcription.dataset_loader import DataSetLoader
from pathlib import Path
import pytest


def test_data_set_loader():
    test_files = Path(__file__).parent / 'test_files' / 'toy_dataset'

    try:
        data_set = DataSetLoader(test_files)
        assert len(data_set.file_names) == 5
        assert data_set.dim_x == (60, 5)
        assert data_set.dim_y == (9,)

    except Exception as e:
        pytest.fail(f"An non intended exception was raised: {e}")


@pytest.mark.run(after='test_data_set_loader')
def test_get_training_validation_file_names():
    test_files = Path(__file__).parent / 'test_files' / 'toy_dataset'

    data_set = DataSetLoader(test_files)
    train_names, validation_names = data_set.get_training_validation_file_names(training_percent=0.8)

    assert len(train_names) == 4
    assert len(validation_names) == 1

    assert set(train_names) != set(validation_names)
