"""DICOM in, pixels + spacing + display PNG out."""

from examples.make_sample_dicom import write_dicom

from anotmed.io_dicom import image_png, load_dicom


def test_load_and_render(tmp_path):
    p = tmp_path / "s.dcm"
    write_dicom(str(p), 128)

    arr, meta = load_dicom(str(p))
    assert arr.shape == (128, 128)
    assert meta.pixel_spacing_mm == (0.7, 0.7)
    assert meta.modality == "CT"
    assert meta.spacing_is_default is False

    png = image_png(arr, meta)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
