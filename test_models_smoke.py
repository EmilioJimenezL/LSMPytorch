"""
Smoke tests — verify all registered models accept the correct input/output shapes.
Run: python test_models_smoke.py
"""

import sys
import torch
from models import create_model, MODEL_REGISTRY


def run_smoke_tests():
    print("=" * 60)
    print("SMOKE TESTS: Model Implementations")
    print("=" * 60)

    num_classes = 330
    batch_size = 4
    num_frames = 85
    feature_dim = 135

    x = torch.randn(batch_size, num_frames, feature_dim)

    all_passed = True

    for model_name in MODEL_REGISTRY.keys():
        print(f"\n{model_name.upper()}")
        print("-" * 60)

        try:
            model = create_model(model_name, num_classes)
            model.eval()

            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"

            params = sum(p.numel() for p in model.parameters())

            assert not torch.isnan(output).any(), "Output contains NaN"

            print(f"  Output shape : {tuple(output.shape)}")
            print(f"  Parameters   : {params:,}")
            print(f"  No NaN values: OK")

            if hasattr(model, 'receptive_field'):
                rf = model.receptive_field()
                print(f"  Receptive field: {rf} frames ({rf / num_frames * 100:.1f}%)")

            print(f"  {model_name} PASSED")

        except Exception as e:
            print(f"  {model_name} FAILED: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SMOKE TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see above")
    print("=" * 60)
    return all_passed


if __name__ == '__main__':
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
