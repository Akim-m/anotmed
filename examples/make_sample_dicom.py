"""Synthesize a small, fully synthetic DICOM for testing.

Two bright blobs on a dark background with realistic-ish CT metadata (pixel
spacing, window). No real patient data is involved. Run:

    python examples/make_sample_dicom.py sample.dcm
"""

from __future__ import annotations

import argparse

import numpy as np


def synth_image(size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = np.full((size, size), 40.0)
    for cy, cx, r, peak in [(90, 100, 14, 900), (170, 165, 9, 820)]:
        img += peak * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
    rng = np.random.default_rng(0)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 4095).astype(np.uint16)


def write_dicom(path: str, size: int = 256) -> None:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    img = synth_image(size)

    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()

    ds = FileDataset(path, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientName = "SYNTHETIC^PHANTOM"  # not real PHI
    ds.PatientID = "SYN0001"

    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.7, 0.7]
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = 400
    ds.WindowWidth = 1200
    ds.PixelData = img.tobytes()

    # Encoding is fixed by file_meta.TransferSyntaxUID (Explicit VR Little
    # Endian). pydicom 3.x takes it as save_as kwargs; 2.x wants the (now
    # deprecated) dataset attributes. Support both so the writer survives a
    # pydicom 4.0 bump without dropping the >=2.4 floor.
    try:
        ds.save_as(path, little_endian=True, implicit_vr=False)  # pydicom >= 3.0
    except TypeError:
        ds.is_little_endian = True  # pydicom 2.x
        ds.is_implicit_VR = False
        ds.save_as(path)
    print(f"wrote {path}  ({size}x{size}, spacing 0.7mm, 2 synthetic lesions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="sample.dcm")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    write_dicom(args.out, args.size)
