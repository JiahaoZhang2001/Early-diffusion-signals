# Data Setup

The current pipeline expects the Twitter15/16 tree dataset under `Data/rumor_detection_acl2017/`.

Minimal required structure:
```text
Data/
└── rumor_detection_acl2017/
    ├── twitter15/
    │   ├── label.txt
    │   └── tree/
    │       ├── <cascade_id>.txt
    │       └── ...
    └── twitter16/
        ├── label.txt
        └── tree/
            ├── <cascade_id>.txt
            └── ...
```

Required files used by `main.py`:
- `Data/rumor_detection_acl2017/twitter15/label.txt`
- `Data/rumor_detection_acl2017/twitter16/label.txt`
- `Data/rumor_detection_acl2017/twitter15/tree/*.txt`
- `Data/rumor_detection_acl2017/twitter16/tree/*.txt`

If you pass `--data_dir Data`, the code automatically uses `Data/rumor_detection_acl2017/` when present.
If your copy is elsewhere, run with `--data_dir <your_data_root>`.
