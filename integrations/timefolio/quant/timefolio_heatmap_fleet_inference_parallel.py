"""A separately registered exact-column parallel replication of full inference."""
import argparse
from pathlib import Path
import time
from quant import timefolio_heatmap_fleet_inference as original
from quant.timefolio_heatmap_fleet_inference_v2 import run
from quant.timefolio_heatmap_parallel_bootstrap import family_bootstrap
from quant.timefolio_heatmap_gpu_worker import digest, write


def replicate(root, output):
    assert not output.exists();output.mkdir(parents=True)
    import quant.timefolio_heatmap_parallel_bootstrap as engine
    write(output/'compute_registration.json',dict(at=time.time(),
        purpose='Reproduce the registered family using concurrent independent columns',
        input_manifest_sha256=digest(root/'input_hashes.json'),workers=8,
        engine_sha256=digest(Path(engine.__file__)),parent_engine_sha256=engine.PARENT_SHA256,
        new_hypotheses=0,new_training=False,unchanged_draws=4000,unchanged_seed=57,
        unchanged_blocks=[5,10],full_reference_comparison_required=True))
    original.family_bootstrap=family_bootstrap
    run(root,output/'evaluation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();replicate(a.root,a.output)
