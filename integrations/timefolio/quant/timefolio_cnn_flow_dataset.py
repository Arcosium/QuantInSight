"""Materialize three matched 22-row datasets without changing active trainers."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from quant.timefolio_cnn_dataset import sha
from quant.timefolio_cnn_flow_context import FEATURES, MODES, compose_images


def verified_file(root, name, digest):
    path = (root/name).resolve()
    if not path.is_relative_to(root) or sha(path) != digest:
        raise ValueError('Registered source artifact changed or escaped its directory')
    return path


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def build(dataset, account_panel, flow_panel, output):
    dataset, account_panel, flow_panel, output = (Path(p).resolve()
        for p in (dataset, account_panel, flow_panel, output))
    if output.exists() or (dataset/'RETIRED.json').exists():
        raise ValueError('A fresh output and nonretired parent are required')
    manifest = json.loads((dataset/'manifest.json').read_text())
    parent_digest = sha(dataset/'manifest.json')
    if manifest.get('feature_rows') != 16 or manifest.get('exploratory_ready') is not True:
        raise ValueError('The existing exploratory 16-row parent is required')
    flow_proof = json.loads((flow_panel/'receipt.json').read_text())
    if (flow_proof['parent_manifest_sha256'] != parent_digest or flow_proof['lag_sessions'] != 1
            or flow_proof['features'] != FEATURES or flow_proof['composed_image_height'] != 22):
        raise ValueError('Registered causal flow panel identity mismatch')
    flow_source = Path(__file__).with_name('timefolio_cnn_flow_context.py')
    rank_source = Path(__file__).with_name('timefolio_cnn_context.py')
    if (sha(flow_source) != flow_proof['source_module_sha256']
            or sha(rank_source) != flow_proof['rank_module_sha256']):
        raise ValueError('Flow preparation source changed')
    arrays = {}
    for name, spec in manifest['arrays'].items():
        path = verified_file(dataset,spec['path'],spec['sha256'])
        arrays[name] = np.load(path,mmap_mode='r',allow_pickle=False)
    for name,digest in flow_proof['artifacts'].items():
        verified_file(flow_panel,name,digest)
    for name in ('dates','codes'):
        np.testing.assert_array_equal(arrays[name],np.load(flow_panel/(name+'.npy'),allow_pickle=False))
    context = np.load(flow_panel/'rows.npy',mmap_mode='r',allow_pickle=False)
    images = arrays['images']
    if (images.dtype != np.float32 or images.ndim != 4 or images.shape[1:3] != (1,16)
            or images.shape[-1] != manifest['window'] or len(images) != manifest['rows']
            or context.shape != (len(arrays['dates']),len(arrays['codes']),6)
            or str(arrays['dates'][-1]) > '20260923'):
        raise ValueError('Registered image and date axes required')
    account = json.loads((account_panel/'receipt.json').read_text())
    if account['dataset_manifest_sha256'] != parent_digest:
        raise ValueError('Account panel must identify the same parent dataset')
    for name,digest in account['artifacts'].items():
        verified_file(account_panel,name,digest)
    index = json.loads((account_panel/'index.json').read_text())
    if index['dates'] != arrays['dates'].tolist() or index['codes'] != arrays['codes'].tolist():
        raise ValueError('Account identity axes differ from image axes')
    expected_bytes = len(MODES)*(len(images)*22*images.shape[-1]*4+4096)
    if shutil.disk_usage(output.parent).free < expected_bytes+10*1024**3:
        raise ValueError('Insufficient free disk with a10GiB operating margin')
    output.mkdir(parents=True)
    started = time.monotonic()
    recipe = dict(parent_manifest_sha256=parent_digest, flow_panel_receipt_sha256=sha(flow_panel/'receipt.json'),
        account_panel_receipt_sha256=sha(account_panel/'receipt.json'), builder_sha256=sha(__file__),
        composer_sha256=sha(flow_source), modes=list(MODES), added_features=FEATURES,
        rows=len(images), image_height=22, window=images.shape[-1],
        estimated_new_image_bytes=expected_bytes, current_frozen_worker_integration_verified=False)
    write(output/'recipe.json',recipe)
    records = []
    for mode in MODES:
        root = output/mode;child = root/'dataset';child.mkdir(parents=True)
        destination = np.lib.format.open_memmap(child/'images.npy',mode='w+',dtype=np.float32,
            shape=(len(images),1,22,images.shape[-1]))
        for lo in range(0,len(images),512):
            hi = min(lo+512,len(images))
            destination[lo:hi] = compose_images(images[lo:hi],arrays['signal_index'][lo:hi],
                arrays['security_key'][lo:hi],context,mode=mode)
            if lo % (512*32) == 0:
                destination.flush()
                write(output/'progress.json',dict(at=time.time(),state='building_images',mode=mode,
                    completed=hi,planned=len(images),completed_modes=len(records)))
        destination.flush();del destination
        for name,spec in manifest['arrays'].items():
            if name != 'images':
                target = child/spec['path'];target.parent.mkdir(parents=True,exist_ok=True)
                os.link(dataset/spec['path'],target)
        for filename in ('cohort.json','price_reference_hashes.json'):
            if (dataset/filename).exists():
                shutil.copy2(dataset/filename,child/filename)
        child_manifest = copy.deepcopy(manifest)
        child_manifest['arrays']['images'] = dict(path='images.npy',sha256=sha(child/'images.npy'))
        child_manifest.update(feature_rows=22,flow_mode=mode,flow_features=FEATURES,
            flow_parent_manifest_sha256=parent_digest,flow_panel_receipt_sha256=recipe['flow_panel_receipt_sha256'],
            flow_builder_sha256=recipe['builder_sha256'],flow_composer_sha256=recipe['composer_sha256'],
            flow_lag_sessions=1,flow_worker_required_image_height=22,
            flow_controls_preserve_prediction_membership=True,flow_controls_preserve_economic_labels=True,
            flow_control_description={'padded':'Six neutral zero rows; no flow values or availability.',
                'missingness':'Four neutral zero rows followed by the same two availability masks.',
                'flow':'Four lagged cumulative net-share ranks and two availability masks.'}[mode],
            limitations=manifest['limitations']+[
                'Partial source coverage and historical publication/revision vintages remain uncertified.',
                'Flow rows contain signed net-share ranks, not net-money or identified pension flows.',
                'This image package requires separately verified22-row worker support before training.'],
            flow_derived_at=time.time())
        write(child/'manifest.json',child_manifest)
        copied_panel = root/'account_panel';copied_panel.mkdir()
        for filename in account['artifacts']:
            target=copied_panel/filename;target.parent.mkdir(parents=True,exist_ok=True)
            os.link(account_panel/filename,target)
        copied_proof = copy.deepcopy(account)
        copied_proof.update(dataset_manifest_sha256=sha(child/'manifest.json'),
            flow_parent_panel_receipt_sha256=recipe['account_panel_receipt_sha256'],
            reused_account_arrays_without_modification=True)
        write(copied_panel/'receipt.json',copied_proof)
        record = dict(mode=mode,manifest_sha256=sha(child/'manifest.json'),
            account_receipt_sha256=sha(copied_panel/'receipt.json'),
            images_sha256=child_manifest['arrays']['images']['sha256'])
        records.append(record)
        write(output/'progress.json',dict(at=time.time(),state='mode_built',completed_modes=len(records),
            planned_modes=len(MODES),last_mode=mode,seconds=time.monotonic()-started))
        print(json.dumps(dict(mode=mode,rows=len(images),image_height=22)),flush=True)
    write(output/'complete.json',dict(at=time.time(),recipe_sha256=sha(output/'recipe.json'),modes=records,
        model_training_started=False,worker22row_integration_verified=False,main_review_pending=True))
    write(output/'progress.json',dict(at=time.time(),state='three_datasets_built_pending_main_review',
        completed_modes=3,rows_per_mode=len(images),seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('dataset','account-panel','flow-panel','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    build(args.dataset,args.account_panel,args.flow_panel,args.output)
