# 使 `import nano_sync_freq` 在 `umi_scripts_csw` 为 cwd 时可用
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent  # nano_sync_freq
_UMI = _PKG.parent  # umi_scripts_csw
if str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))
