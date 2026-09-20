#!/usr/bin/env python
"""Resume a matching draft release with bounded independent asset uploads.

Already uploaded assets must match local size and SHA-256. Conflicting assets
are never overwritten or deleted. Publication still requires the exact full
asset-set verification performed by publish_assessment.verify_remote.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
from publish_assessment import GH,REPO,release_files,verify_remote
from rna import hash_file


def missing_assets(expected,remote):
    if set(remote)-set(expected):raise ValueError('unexpected_remote_release_assets')
    missing=[]
    for name,item in expected.items():
        asset=remote.get(name)
        if asset is None:missing.append(name);continue
        if asset['size']!=item['size'] or asset.get('digest')!='sha256:'+item['sha256'] or asset.get('state')!='uploaded':
            raise ValueError('existing_remote_asset_not_verified:'+name)
    return missing


def resume(output,tag,workers):
    files=release_files(output);identity=json.loads((output/'release-identity.json').read_text())
    expected={p.name:{'size':p.stat().st_size,'sha256':hash_file(p)} for p in files}
    pages=json.loads(subprocess.check_output([GH,'api',f'repos/{REPO}/releases?per_page=100','--paginate','--slurp'],text=True))
    matches=[r for page in pages for r in page if r['tag_name']==tag]
    if len(matches)!=1:raise ValueError('release_identity_missing_or_ambiguous')
    release=matches[0]
    if release['target_commitish']!=identity['publisher_commit'] or identity['bundle_id'] not in release['body']:raise ValueError('draft_release_scope_changed')
    pages=json.loads(subprocess.check_output([GH,'api',f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100','--paginate','--slurp'],text=True))
    remaining=missing_assets(expected,{a['name']:a for page in pages for a in page})
    if remaining and not release['draft']:raise ValueError('cannot_add_assets_to_published_assessment')
    print(json.dumps({'verified_existing':len(files)-len(remaining),'remaining':remaining,'workers':workers}),flush=True)
    def upload(name):
        subprocess.run([GH,'release','upload',tag,str(output/name),'--repo',REPO],check=True)
        print('uploaded '+name,flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(upload,name) for name in remaining]
        for job in as_completed(jobs):job.result()
    return verify_remote(output,tag)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('workers must be between 1 and 4')
    print(resume(a.output,a.tag,a.workers),flush=True)
