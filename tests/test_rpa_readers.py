import json

from rpa_readers import read_app_config


def test_read_app_config_uses_selected_model_metadata(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "onmyoji.yaml").write_text("names: ['default_a', 'default_b']\n", encoding="utf-8")
    (data_dir / "onmyoji_name.yaml").write_text(
        "Class_Name_To_Chinese: {default_a: '默认A', default_b: '默认B'}\n",
        encoding="utf-8",
    )

    model_path = tmp_path / "custom_model.pt"
    model_path.write_bytes(b"fake")
    (tmp_path / "custom_model.yaml").write_text("names: ['alpha', 'beta']\n", encoding="utf-8")
    (tmp_path / "settings.json").write_text(
        json.dumps({"model_path": str(model_path)}),
        encoding="utf-8",
    )

    config = read_app_config(root=tmp_path)

    assert config.model_path == str(model_path)
    assert config.class_names == ("alpha", "beta")
