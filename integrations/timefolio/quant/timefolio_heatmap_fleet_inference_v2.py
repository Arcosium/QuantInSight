"""Preserve registered inference while applying the user's stricter stop rule."""
import argparse
import json
from pathlib import Path
from quant.timefolio_heatmap_fleet_inference import run as registered_inference
from quant.timefolio_heatmap_fleet_lab import cases
from quant.timefolio_heatmap_fleet_thresholds import classify
from quant.timefolio_heatmap_gpu_worker import write, digest


def run(root, output):
    assert not output.exists(); output.mkdir(parents=True)
    # Original comparisons and joint max-t family remain unchanged. Its old
    # 3/1 diagnostic is retained as historical evidence, never the current gate.
    registered_inference(root, output / 'registered_inference')
    original = json.loads((output / 'registered_inference/evaluation_summary.json').read_text())
    retained, stop = [], []
    for cfg in cases():
        source = root / 'results' / cfg['id']
        for row in json.loads((source / 'paper_proximity.json').read_text()):
            verdict = classify(row['monthly_target'])
            record = dict(id=row['id'], case=cfg['id'], **verdict, monthly=row['monthly_target'])
            if verdict['retain_candidate']: retained.append(record)
            if verdict['stop_search_and_validate']: stop.append(record)
    write(output / 'retained_candidates.json', retained)
    write(output / 'stop_candidates.json', stop)
    write(output / 'evaluation_summary.json', dict(status='main_review_pending',
        portfolios=19984, joint_hypotheses=122096, stop_rule='pooled>3 and every monthly fold>1.5',
        record_rule='pooled>2 and every monthly fold>1', all_members_retained=True,
        retained_accounts=len(retained), stop_accounts=len(stop),
        primary_ensemble_statistical_gate_passed=original['robust_candidate_gate_passed'],
        current_target_and_primary_statistical_gate_passed=sorted(
            {r['id'] for r in stop} & set(original['robust_candidate_gate_passed'])),
        old_target_diagnostic_location='registered_inference/evaluation_summary.json',
        reused_development_only=True, independent_confirmation=False,
        full_contest_compliance_certified=False, reserved_outcomes_read=False, orders_submitted=False))
    write(output / 'artifact_hashes.json', {str(p.relative_to(output)): digest(p)
          for p in output.rglob('*') if p.is_file()})
    write(output / 'complete.json', dict(status='main_review_pending',
        input_manifest_sha256=digest(root / 'input_hashes.json'), source_sha256=digest(Path(__file__))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.root,args.output)
