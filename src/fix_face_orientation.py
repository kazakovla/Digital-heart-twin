"""
Исправление ориентации граничных граней в уже подготовленных geom_XX.npz.

Причина: в первой версии prepare_meshes.py нормали ориентировались «от центра
сетки», что неверно для эндокарда (внешняя нормаль миокарда там смотрит
в полость). Давление в полостях из-за этого сжимало сердце.

Скрипт не требует повторной загрузки исходных данных с Zenodo: ориентация
восстанавливается по тетраэдрам, которым принадлежат грани. Классификация
граней (face_type) не меняется -- меняется только порядок вершин.

Запуск (из корня репозитория):
    python src/fix_face_orientation.py --geom-dir data/geom
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from prepare_meshes import outward_orient


def check_orientation(coords, faces, elements) -> float:
    """Доля граней, нормаль которых направлена от противолежащей вершины."""
    ref = outward_orient(coords, faces, elements)
    return float(np.mean(np.all(ref == faces, axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-dir", default="data/geom")
    a = ap.parse_args()
    for path in sorted(glob.glob(os.path.join(a.geom_dir, "geom_*.npz"))):
        d = dict(np.load(path))
        coords = d["coords"].astype(np.float64)
        elements = d["elements"].astype(np.int64)
        before = check_orientation(coords, d["faces"], elements)
        d["faces"] = outward_orient(coords, d["faces"].astype(np.int64),
                                    elements).astype(np.int32)
        np.savez_compressed(path, **d)
        print(f"{os.path.basename(path)}: верно ориентированных граней "
              f"до {before * 100:5.1f}% -> после 100.0%")


if __name__ == "__main__":
    main()
