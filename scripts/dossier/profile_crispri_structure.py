#!/usr/bin/env python
"""Audit complete Replogle single-cell and pseudobulk inputs without mutation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
from crispri import REPLOGLE, task_metadata, control_coverage, bulk_comparison
from rna import RNAFile, hash_file, scan, mapping_audit, value_hash
from profile_responses import write_json, serial
from render import render

ROOT = Path(__file__).resolve().parents[2]


def assess(inventory, evidence, output):
    if output.exists() or output.resolve().is_relative_to((ROOT / 'data/raw').resolve()):
        raise ValueError('fresh_non_raw_output_required')
    inv = json.loads(inventory.read_text())
    if inv['status'] != 'completed':
        raise ValueError('completed_inventory_required')
    files = {Path(f['file']).name: f['sha256'] for f in inv['file_results'] if f['source_id'] == 'replogle2022'}
    if set(files) != {f'{c}_raw_{kind}_01.h5ad' for c in REPLOGLE for kind in ['singlecell','bulk']}:
        raise ValueError('unexpected_Replogle_input_scope')
    hgnc_path = ROOT / 'data/raw/networks/hgnc_complete_set.txt'
    axis_path = ROOT / 'data/raw/arc_vcc2026_controls/gene_names.csv'
    input_paths = [ROOT / 'data/raw/replogle2022' / n for n in files] + [hgnc_path, axis_path]
    hashes = {str(p.relative_to(ROOT)): hash_file(p) for p in input_paths}
    for p in input_paths:
        if p.name in files and hashes[str(p.relative_to(ROOT))] != files[p.name]:
            raise ValueError('source_inventory_hash_mismatch')
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in input_paths}
    for record in json.loads((evidence / 'evidence.json').read_text()):
        if record.get('file') and hash_file(evidence / record['file']) != record['sha256']:
            raise ValueError('evidence_hash_mismatch')
    output.mkdir(parents=True)
    shutil.copytree(evidence, output / 'evidence')
    t0 = time.monotonic()
    hgnc = pd.read_csv(hgnc_path, sep='\t', low_memory=False)
    official = pd.read_csv(axis_path).gene_name.tolist()
    summaries, bulk_summaries, overlaps, axes, experiments = [], [], [], [], {}
    for experiment, context in REPLOGLE.items():
        directory = output / experiment
        directory.mkdir()
        path = ROOT / 'data/raw/replogle2022' / f'{experiment}_raw_singlecell_01.h5ad'
        bulk = path.with_name(f'{experiment}_raw_bulk_01.h5ad')
        digest = hashes[str(path.relative_to(ROOT))]
        print(f'scan {experiment}', flush=True)
        summary, cells, genes = scan(path, digest, 'Replogle2022:' + experiment)
        if not summary['all_counts_finite_nonnegative_integer']:
            raise ValueError('singlecell_raw_count_semantics_failed')
        with RNAFile(path) as source:
            obs = source.obs
            tasks = task_metadata(obs)
            coverage = control_coverage(obs)
            cells['source_batch'] = obs.gem_group.astype(str).to_numpy()
            cells['source_target_gene'] = obs.gene.to_numpy()
            cells['source_guide_id'] = obs.sgID_AB.to_numpy()
            cells['source_task'] = obs.gene_transcript.to_numpy()
            cells['source_context'] = experiment
            cells['source_cell_line'] = context['cell_line']
            cells['source_days_post_transduction'] = context['days_post_transduction']
            manifest_name = {'K562_essential':'KD6', 'K562_gwps':'KD8', 'rpe1':'RD7'}[experiment] + '_raw_files.csv'
            manifest = pd.read_csv(evidence / manifest_name)
            if manifest.gemgroup.duplicated().any():
                raise ValueError('ambiguous_GEM_library_mapping')
            libraries = manifest.set_index('gemgroup').library
            cells['source_library'] = obs.gem_group.map(libraries).to_numpy()
            if cells.source_library.isna().any():
                raise ValueError('GEM_missing_from_primary_library_manifest')
            cells['independent_biological_replicate'] = None
            cells['computed_identity_key'] = [value_hash(['Replogle2022', experiment, str(g), str(b)]) for g,b in zip(obs.gem_group, obs.index)]
            experiments[experiment] = {'targets': set(obs.gene) - {'non-targeting'}, 'constructs': set(obs.gene_transcript),
                'barcodes': set(obs.index), 'source_axis': list(source.var.index)}
        mapping = mapping_audit(genes.gene_name.tolist(), hgnc, official, genes.source_gene.tolist())
        mapping.insert(0, 'source_gene_id', genes.source_gene.tolist())
        for gene in official:
            matches = genes.source_gene[mapping.mapped_symbol == gene].tolist()
            axes.append({'context': experiment, 'official_gene': gene, 'measured_native_rows': len(matches),
                         'native_gene_ids': '|'.join(matches), 'measured': bool(matches),
                         'unmeasured_is_not_zero': not bool(matches)})
        ids = cells.source_target_gene == 'non-targeting'
        summary.update(context=experiment, **context, n_NTC_cells=int(ids.sum()),
            perturbation_construct_tasks=int((~tasks.is_control).sum()), target_genes=len(experiments[experiment]['targets']),
            guide_component_counts=tasks.guide_components.value_counts().to_dict(),
            GEM_groups=cells.source_batch.nunique(), unmatched_task_GEM_rows=int((~coverage.matched & ~coverage.is_control).sum()),
            gene_mapping_status=mapping.mapping_status.value_counts().to_dict(),
            symbol_ensembl_conflicts=int((mapping.symbol_vs_ensembl == 'conflict').sum()),
            independent_biological_replicates=None,
            exact_count_duplicates_within_context=int(cells.computed_count_sha256.duplicated().sum()))
        for name, frame in [('cells', cells), ('genes', genes), ('gene_mapping', mapping), ('tasks', tasks), ('control_coverage', coverage)]:
            frame.to_parquet(directory / f'{name}.parquet', index=False, compression='zstd')
        print(f'bulk comparison {experiment}', flush=True)
        comparison, bulk_summary = bulk_comparison(path, bulk)
        comparison.to_parquet(directory / 'bulk_comparison.parquet', index=False)
        with RNAFile(bulk) as source:
            source.obs.rename_axis('source_task').reset_index().add_prefix('source_').to_parquet(directory / 'bulk_source_metadata.parquet', index=False)
        write_json(directory / 'summary.json', summary)
        summaries.append(summary)
        bulk_summaries.append(dict(bulk_summary, context=experiment))
        print(f'completed structure {experiment}: {len(cells)} cells, {summary["perturbation_construct_tasks"]} tasks', flush=True)
    for i, a in enumerate(experiments):
        for b in list(experiments)[:i]:
            aa, bb = experiments[a], experiments[b]
            overlaps.append({'context_A': a, 'context_B': b, 'shared_target_genes': len(aa['targets'] & bb['targets']),
                'shared_construct_labels': len(aa['constructs'] & bb['constructs']), 'shared_barcode_strings': len(aa['barcodes'] & bb['barcodes']),
                'shared_measured_Ensembl_genes': len(set(aa['source_axis']) & set(bb['source_axis'])),
                'same_experiment': False, 'deduplication_status': 'not_duplicate_evidence',
                'reason': 'Different source experiments/timepoints; 10x barcode namespaces are reusable. No cells removed.'})
    pd.DataFrame(axes).to_parquet(output / 'official_axis_coverage.parquet', index=False)
    changed = [p for p,v in before.items() if v != (Path(p).stat().st_size, Path(p).stat().st_mtime_ns, Path(p).stat().st_ctime_ns)]
    report = {'schema_version': 2, 'bundle_id': 'replogle-structure-' + uuid.uuid4().hex,
        'title': 'Replogle 全量来源、构件和测量语义核查', 'status': 'completed' if not changed else 'failed',
        'completed_at': datetime.now(timezone.utc).isoformat(), 'duration_seconds': time.monotonic() - t0,
        'input_sha256': hashes, 'inventory_sha256': hash_file(inventory), 'inputs_unchanged': not changed,
        'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'code': {n: hash_file(Path(__file__).with_name(n)) for n in ['crispri.py','rna.py','profile_crispri_structure.py','profile_responses.py','render.py']},
        'runtime': {'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in ['numpy','scipy','pandas','h5py','pyarrow']},
                    'uv_lock_sha256': hash_file(Path(__file__).with_name('uv.lock'))},
        'methods': {'protocol': 'https://github.com/yjcyxky/virtual-cell-challenge/issues/7',
            'identity': 'SHA256 input + source row, plus study/experiment/GEM/barcode index; no cross-experiment barcode-only deduplication',
            'bulk': 'All source rows numeric audit and all local single-cell group means compared at float32 tolerance; not independent experiments',
            'mapping': 'Source symbols and Ensembl checked against fixed local HGNC; native axis preserved; official unmeasured entries are not zero'},
        'limitations': ['本阶段完成工程及设计核查；全任务响应与逐细胞推断由后续计算产物提供。',
            'gene_transcript 保留启动子／转录本差异；双 guide 为同一构件，GEM 分组不是独立培养重复。',
            '来源只保留平均表达 >0.01 UMI/cell 的基因；未测量的官方基因不能填为真实零值。',
            '原始标签、pseudobulk 的源 DE 列与本轮计算分开保存；来源 raw 命名不保证整数语义。'],
        'tables': [{'title':'全部单细胞输入','rows':summaries}, {'title':'全部 pseudobulk 输入','rows':bulk_summaries},
                   {'title':'实验间覆盖与身份','rows':overlaps}], 'reproduce': sys.argv}
    report['artifacts'] = [{'file':str(p.relative_to(output)), 'sha256':hash_file(p)} for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'report.json', report)
    (output/'report.html').write_text(render(serial(report)))
    (output/'SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.relative_to(output)}\n' for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','evidence','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assess(args.inventory,args.evidence,args.output)
    except Exception as exc:
        if args.output.exists() and not (args.output/'report.json').exists():
            write_json(args.output/'failure.json', {'status':'failed','error':f'{type(exc).__name__}: {exc}'})
        raise
    print(json.dumps({'status':result['status'],'bundle_id':result['bundle_id']}))
