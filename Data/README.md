# Data Setup

The pipeline expects `--data_dir` to point to a folder containing the Twitter15/16 tree dataset.

Default command in this repo:
```bash
python3 main.py --data_dir Data --out_dir thesis_outputs
```

With this default, the script automatically looks for:
- `Data/rumor_detection_acl2017/twitter15/`
- `Data/rumor_detection_acl2017/twitter16/`

Required files used by the current analysis:
- `Data/rumor_detection_acl2017/twitter15/label.txt`
- `Data/rumor_detection_acl2017/twitter16/label.txt`
- `Data/rumor_detection_acl2017/twitter15/tree/*.txt`
- `Data/rumor_detection_acl2017/twitter16/tree/*.txt`

If your data is stored elsewhere, pass a different `--data_dir`.
The code also supports pointing directly to the `rumor_detection_acl2017` folder.
