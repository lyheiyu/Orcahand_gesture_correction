# Chinese Dance Gesture Dataset Release

## Canonical paper dataset

- Release asset: `chinese_dance_6class_dataset_v1.zip`
- Uncompressed file: `gesture_sequence_dataset_chinese_dance_6class_after_fix.csv`
- Gesture classes: 6
- Sequences: 571
- Frames: 26,260
- Columns: 221
- ZIP size: 41.88 MiB
- ZIP SHA-256: `441702C3357CC7C5BE93EB8B914DCD5B1174EFF31229E8C5D60CAFFE60206A4A`

The archive contains the six-class dataset used by `main.tex`. It does not
contain the numerical gesture labels 6, 7, and 8. Conventional smoothing
features and other intermediate experiment files are intentionally excluded
because they can be regenerated from the canonical dataset using the scripts
in this repository.

## Publishing on GitHub

1. Push the source-code commit to GitHub.
2. Open the repository's **Releases** page.
3. Choose **Draft a new release**.
4. Create tag `dataset-v1` and title the release `Chinese Dance Gesture Dataset v1`.
5. Upload `release_assets/chinese_dance_6class_dataset_v1.zip` as a release asset.
6. Publish the release and add its public URL to this document.

Release URL: `TODO_AFTER_UPLOAD`

## Integrity check

After downloading the archive on Windows, verify it with:

```powershell
Get-FileHash -Algorithm SHA256 .\chinese_dance_6class_dataset_v1.zip
```

The returned hash must match the SHA-256 value above.
