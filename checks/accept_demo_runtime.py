#!/usr/bin/env python3
"""Opt-in CPU acceptance of archived validation examples; never run by CI.

Run in an isolated runtime and an OS network-denial policy. This is not a
formal A-E evaluation and cannot read a dataset or generate test predictions.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Acceptance receipt already exists'
    root = args.demo.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    from PIL import Image
    import onnxruntime as ort
    import predict

    checks = []
    errors = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append({'name': name, 'status': 'passed', 'detail': detail})
        except Exception as exc:
            errors.append({'name': name, 'error': repr(exc)})
            checks.append({'name': name, 'status': 'failed'})

    def expect_demo_error(fn):
        try:
            fn()
        except predict.DemoError as exc:
            return str(exc)
        raise AssertionError('Expected explicit DemoError rejection')

    def network_probe():
        try:
            with socket.create_connection(('1.1.1.1', 443), timeout=0.5):
                pass
        except PermissionError as exc:
            return {'os_denied': True, 'errno': exc.errno, 'endpoint': '1.1.1.1:443'}
        raise AssertionError('OS network denial was not demonstrated')

    check('external_network_denied_by_os', network_probe)
    check('no_training_dependencies', lambda: {
        name: assert_absent(name) for name in ('torch', 'torchvision', 'medmnist', 'sklearn')
    })
    predictor = predict.Predictor()
    manifest = predict.load_samples()

    def fixed_samples():
        result = []
        for sample in manifest['samples']:
            assert sample['sample_id'].startswith('val:')
            out = predictor.predict((root/'samples'/sample['file']).read_bytes())
            delta = float(np.max(np.abs(np.asarray(out['probabilities']) - sample['reference_probabilities'])))
            assert delta <= 1e-5, (sample['sample_id'], delta)
            assert out['predicted_label'] == sample['reference_prediction']
            result.append({'sample_id': sample['sample_id'], 'given_label': sample['given_label'],
                           'predicted_label': out['predicted_label'], 'max_abs_probability_error': delta})
        return {'samples': result, 'absolute_probability_tolerance': 1e-5,
                'correct_example': 'val:0', 'fixed_disagreement': 'val:9082'}

    check('ten_archived_validation_samples_match_reference', fixed_samples)
    for mode in ('L', 'RGBA'):
        def conversion(mode=mode):
            image = Image.open(root/'samples/val_0.png').convert(mode)
            buf = io.BytesIO(); image.save(buf, format='PNG')
            out = predictor.predict(buf.getvalue())
            assert out['input_mode'] == mode and len(out['probabilities']) == 9
            return {'input_mode': mode, 'finite_nine_class_output': True,
                    'fixture': 'format-converted validation image, not clinical evidence'}
        check('input_conversion_'+mode, conversion)
    check('corrupt_input_rejected', lambda: expect_demo_error(lambda: predictor.predict(b'not an image')))

    with tempfile.TemporaryDirectory(prefix='metadata_acceptance_') as temp:
        model_dir = Path(temp)/'model'; shutil.copytree(root/'model', model_dir)
        original = (model_dir/'metadata.json').read_bytes()
        for key, value in [('labels', list(reversed(predict.LABELS))), ('size', 64),
                           ('dtype', 'float16'), ('layout', 'NHWC'), ('mean', [0., 0., 0.]),
                           ('std', [1., 1., 1.]), ('variant', 'int8')]:
            def reject_metadata(key=key, value=value):
                m = json.loads(original); m[key] = value
                (model_dir/'metadata.json').write_text(json.dumps(m))
                return expect_demo_error(lambda: predict.validate_metadata(model_dir))
            check('metadata_mismatch_'+key, reject_metadata)
        for path in ('/tmp/fp32.onnx', '../fp32.onnx'):
            def reject_path(path=path):
                m = json.loads(original); m['model']['path'] = path
                (model_dir/'metadata.json').write_text(json.dumps(m))
                return expect_demo_error(lambda: predict.validate_metadata(model_dir))
            check('model_path_rejected_'+('absolute' if path.startswith('/') else 'parent'), reject_path)
        (model_dir/'metadata.json').write_bytes(original)
        data = (model_dir/'fp32.onnx').read_bytes()
        (model_dir/'fp32.onnx').write_bytes(data[:-1]+bytes([data[-1]^1]))
        check('model_bytes_hash_mismatch', lambda: expect_demo_error(lambda: predict.validate_metadata(model_dir)))
        (model_dir/'fp32.onnx').unlink()
        (model_dir/'fp32.onnx').symlink_to(root/'model/fp32.onnx')
        check('model_symlink_escape_rejected', lambda: expect_demo_error(lambda: predict.validate_metadata(model_dir)))

    with tempfile.TemporaryDirectory(prefix='cli_acceptance_') as temp:
        out = Path(temp)/'prediction.json'
        cli = [sys.executable, str(root/'predict.py')]
        first = subprocess.run(cli+['--output', str(out)], cwd=temp, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr+first.stdout
        saved = out.read_bytes()
        check('default_cli_correct_sample', lambda: assert_equal(json.loads(saved)['predicted_label'], 5))
        second = subprocess.run(cli+['--output', str(out)], cwd=temp, capture_output=True, text=True)
        check('cli_refuses_output_overwrite', lambda: assert_equal(
            (second.returncode, out.read_bytes(), '拒绝覆盖' in second.stdout), (2, saved, True)))
        disagreement = subprocess.run(cli+[str(root/'samples/val_9082.png')], cwd=temp, capture_output=True, text=True)
        check('cli_fixed_disagreement', lambda: assert_equal(
            (disagreement.returncode, json.loads(disagreement.stdout)['predicted_label']),
            (0, next(s['reference_prediction'] for s in manifest['samples'] if s['sample_id']=='val:9082'))))

    result = {'status': 'passed' if not errors else 'failed', 'platform': platform.platform(),
              'machine': platform.machine(), 'python': sys.version, 'onnxruntime': ort.__version__,
              'numpy': np.__version__, 'cwd': str(Path.cwd()), 'demo_location': str(root),
              'model_sha256': hashlib.sha256((root/'model/fp32.onnx').read_bytes()).hexdigest(),
              'checks': checks, 'errors': errors, 'model_forward_calls_in_shared_predictor': predictor.calls,
              'additional_cli_forward_calls': 2, 'official_test_inference': 0,
              'new_training_runs': 0, 'browser_render_acceptance': 'separate_receipt',
              'scope': 'new relocated copy, archived validation samples and negative/format fixtures only'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'errors': errors}, ensure_ascii=False))
    return bool(errors)


def assert_absent(name):
    assert importlib.util.find_spec(name) is None, name+' unexpectedly installed'
    return 'absent'


def assert_equal(a, b):
    assert a == b, (a, b)
    return {'matched': True}


if __name__ == '__main__':
    raise SystemExit(main())
