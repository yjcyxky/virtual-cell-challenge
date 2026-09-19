#!/usr/bin/env python
"""Verify, split-package and publish an immutable assessment as a GitHub release."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from rna import hash_file
from profile_responses import write_json

GH='/home/jy001/micromamba/envs/llm-ft/bin/gh'
REPO='yjcyxky/virtual-cell-challenge'


class SplitWriter:
    def __init__(self,directory,limit=1800000000):
        self.directory=directory;self.limit=limit;self.file=None;self.size=0;self.parts=[]

    def write(self,data):
        length=len(data);position=0
        while position<length:
            if self.file is None or self.size==self.limit:
                self.close();path=self.directory/f'assessment.tar.gz.part{len(self.parts)+1:04d}'
                self.file=path.open('xb');self.parts.append(path);self.size=0
            count=min(length-position,self.limit-self.size)
            self.file.write(data[position:position+count]);self.size+=count;position+=count
        return length

    def flush(self):
        if self.file:self.file.flush()

    def close(self):
        if self.file:self.file.close();self.file=None


def verified_files(bundle):
    files=[]
    for line in (bundle/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);path=Path(name)
        if path.is_absolute() or '..' in path.parts or (bundle/path).is_symlink():raise ValueError('unsafe_artifact_path')
        if hash_file(bundle/path)!=digest:raise ValueError('artifact_hash_mismatch: '+name)
        files.append(name)
    if len(set(files))!=len(files) or not {'report.json','report.html'}<=set(files):raise ValueError('invalid_bundle_manifest')
    actual={str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    if actual!=set(files):raise ValueError('unregistered_or_missing_artifacts')
    return files+[str(p.relative_to(bundle)) for p in bundle.rglob('SHA256SUMS')]


def package(bundle,output):
    report=json.loads((bundle/'report.json').read_text())
    if report['status']!='completed':raise ValueError('completed_assessment_required')
    files=verified_files(bundle)
    output.mkdir(parents=True,exist_ok=False);split=SplitWriter(output)
    with gzip.GzipFile(filename='',mode='wb',fileobj=split,compresslevel=1,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as archive:
            for name in sorted(files):archive.add(bundle/name,arcname=name,recursive=False)
    split.close()
    for name in ['report.html','report.json','SHA256SUMS']:shutil.copy2(bundle/name,output/name)
    identity={'bundle_id':report['bundle_id'],'bundle_report_sha256':hash_file(bundle/'report.json'),
        'publisher_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'publisher_sha256':hash_file(Path(__file__)),
        'assets':[{'name':p.name,'size':p.stat().st_size,'sha256':hash_file(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output/'release-identity.json',identity)
    (output/'RELEASE-SHA256SUMS').write_text(''.join(f'{hash_file(p)}  {p.name}\n' for p in sorted(output.iterdir()) if p.is_file()))
    return identity


def release_files(output):
    files=[]
    for line in (output/'RELEASE-SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if Path(name).name!=name:raise ValueError('unsafe_release_asset_name')
        if hash_file(output/name)!=digest:raise ValueError('release_asset_changed')
        files.append(output/name)
    return sorted(files+[output/'RELEASE-SHA256SUMS'])


def verify_remote(output,tag):
    files=release_files(output)
    pages=json.loads(subprocess.check_output([GH,'api',f'repos/{REPO}/releases?per_page=100','--paginate','--slurp'],text=True))
    matches=[r for page in pages for r in page if r['tag_name']==tag]
    if len(matches)!=1:raise ValueError('release_tag_identity_missing_or_ambiguous')
    release=matches[0]
    pages=json.loads(subprocess.check_output([GH,'api',f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100','--paginate','--slurp'],text=True))
    remote={a['name']:a for page in pages for a in page}
    if set(remote)!=set(p.name for p in files):raise ValueError('remote_release_asset_set_mismatch')
    checked=[]
    for path in files:
        a=remote[path.name];digest=hash_file(path)
        if a['size']!=path.stat().st_size or a.get('digest')!='sha256:'+digest:raise ValueError('remote_asset_digest_or_size_mismatch')
        checked.append({'name':path.name,'size':a['size'],'sha256':digest,'remote_digest_verified':True})
    if release['draft']:subprocess.run([GH,'release','edit',tag,'--repo',REPO,'--draft=false'],check=True)
    published=json.loads(subprocess.check_output([GH,'api',f'repos/{REPO}/releases/tags/{tag}'],text=True))
    if published['id']!=release['id'] or published['draft']:raise ValueError('published_release_identity_mismatch')
    write_json(output/'remote-verification.json',{'release_url':published['html_url'],'assets':checked,'status':'completed','verifier_sha256':hash_file(Path(__file__))})
    return published['html_url']


def publish(output,tag,title,notes):
    identity=json.loads((output/'release-identity.json').read_text());files=release_files(output)
    body=notes.read_text()+f'\n\nBundle: `{identity["bundle_id"]}`.\n\n下载全部 `assessment.tar.gz.part*` 后，按文件名顺序连接解包：\n\n```bash\nsha256sum -c RELEASE-SHA256SUMS\nmkdir assessment\ncat assessment.tar.gz.part* | tar -xz -C assessment\ncd assessment\nsha256sum -c SHA256SUMS\n```\n\n打开解包后的 `report.html`。单独下载的 HTML 仅提供概览；关联 sidecars 和子页面位于完整归档中。原始数据不包含在发布包中。\n'
    temporary=output/'release-notes.txt';temporary.write_text(body)
    try:
        subprocess.run([GH,'release','create',tag,'--repo',REPO,'--target',identity['publisher_commit'],'--title',title,'--notes-file',str(temporary),'--draft'],check=True)
    finally:temporary.unlink()
    for path in files:
        subprocess.run([GH,'release','upload',tag,str(path),'--repo',REPO],check=True)
        print('uploaded '+path.name,flush=True)
    return verify_remote(output,tag)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--publish',action='store_true');p.add_argument('--finalize',action='store_true');p.add_argument('--tag');p.add_argument('--title');p.add_argument('--notes',type=Path)
    a=p.parse_args()
    if a.bundle:print(json.dumps(package(a.bundle,a.output)),flush=True)
    if a.publish:
        if not all([a.tag,a.title,a.notes]):p.error('publishing requires tag, title and notes')
        print(publish(a.output,a.tag,a.title,a.notes),flush=True)
    elif a.finalize:
        if not a.tag:p.error('finalizing requires tag')
        print(verify_remote(a.output,a.tag),flush=True)
