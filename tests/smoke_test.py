"""
smoke_test.py
-------------
Runs the full pipeline on procedurally generated synthetic scenes (tests/synthetic.py)
and on basic edge cases (tiny/corrupt/grayscale/RGBA) to confirm the app works end to
end. This is NOT a dataset and NOT used for training or accuracy claims -- it only
exercises the code path.
Run:  python tests/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

import synthetic
from src.preprocessing import load_image, ImageValidationError
from src.pipeline import analyze, run_pipeline


def encode(rgb):
    return cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


def main():
    ok = True
    print("== synthetic scenes ==")
    for name, rgb in synthetic.all_scenes().items():
        out = run_pipeline(encode(rgb), f"{name}.png")
        r = out["result"]
        print(f"{name:10s} -> {r['predicted_class']:11s} conf={r['confidence']:.2f}")

    print("\n== determinism ==")
    rgb = synthetic.forest(1)
    a = run_pipeline(encode(rgb), "f.png")["result"]["scores"]
    b = run_pipeline(encode(rgb), "f.png")["result"]["scores"]
    assert a == b, "classifier is not deterministic!"
    print("deterministic: OK")

    print("\n== edge cases ==")
    try:
        run_pipeline(cv2.imencode(".png", np.zeros((5, 5, 3), np.uint8))[1].tobytes(), "tiny.png")
        ok = False; print("FAIL: tiny image should be rejected")
    except ImageValidationError:
        print("tiny image rejected: OK")
    try:
        run_pipeline(b"garbage", "bad.png")
        ok = False; print("FAIL: corrupt file should be rejected")
    except ImageValidationError:
        print("corrupt file rejected: OK")

    gray = cv2.cvtColor(np.full((200, 200), 120, np.uint8), cv2.COLOR_GRAY2BGR)
    run_pipeline(cv2.imencode(".png", gray)[1].tobytes(), "gray.png")
    print("grayscale image handled: OK")

    rgba = np.random.default_rng(0).integers(0, 255, (200, 200, 4), dtype=np.uint8)
    run_pipeline(cv2.imencode(".png", rgba)[1].tobytes(), "rgba.png")
    print("RGBA image handled: OK")

    print("\nALL SMOKE TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
