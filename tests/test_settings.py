from drum_transcription.settings.settings import Settings


def test_settings_loads():

    try:
        settings = Settings
        print(settings)
    except e:
        pytest.fail("The settings configuration could not be properly loaded")
