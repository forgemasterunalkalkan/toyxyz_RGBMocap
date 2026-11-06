"""Derive the Linux runtime lock from the reviewed Windows pins.

Every group keeps the exact name==version pins of installer/runtime-lock.json;
only the wheel file changes to its manylinux build. Packages that exist only
because of Linux markers (torch's CUDA libraries and Triton) are added with the
versions torch's own metadata pins, and are flagged ``linux_only``.

Developer tool: needs network access and the ``packaging`` library. The
generated installer/runtime-lock-linux.json is what tools/setup_linux.py uses.
"""
from __future__ import annotations
import argparse,hashlib,html,json,re,sys,urllib.request
from pathlib import Path
from packaging import tags
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name,parse_wheel_filename

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'installer/runtime-lock.json'
TARGET=ROOT/'installer/runtime-lock-linux.json'
GLIBC=(2,36)  # Oldest supported host: Debian 12. Newer wheels would not load there.
TORCH_INDEX='https://download.pytorch.org/whl/cu128/'
UV_VERSION='0.13.1'
BLENDER='https://download.blender.org/release/Blender4.5/blender-4.5.0-linux-x64.tar.xz'  # Same release as Windows.
_cache={}


def fetch(url):
    if url not in _cache:
        request=urllib.request.Request(url,headers={'User-Agent':'rgbmocap-linux-lock'})
        with urllib.request.urlopen(request,timeout=60) as response:_cache[url]=response.read()
    return _cache[url]


def supported_tags(python):
    major,minor=map(int,python.split('.')[:2])
    legacy={17:'manylinux2014_x86_64',12:'manylinux2010_x86_64',5:'manylinux1_x86_64'}
    platforms=[p for m in range(GLIBC[1],4,-1) for p in [f'manylinux_2_{m}_x86_64',legacy.get(m)] if p]
    order=list(tags.cpython_tags((major,minor),platforms=platforms))
    order+=list(tags.compatible_tags((major,minor),f'cp{major}{minor}',platforms))
    return {tag:index for index,tag in enumerate(order)}


def environment(python,system):
    linux=system=='linux'
    return dict(python_version='.'.join(python.split('.')[:2]),python_full_version=python,
        os_name='posix' if linux else 'nt',sys_platform='linux' if linux else 'win32',
        platform_system='Linux' if linux else 'Windows',platform_machine='x86_64' if linux else 'AMD64',
        platform_python_implementation='CPython',implementation_name='cpython',
        implementation_version=python,platform_release='',platform_version='',extra='')


def torch_candidates(name,version):
    page=fetch(TORCH_INDEX+canonicalize_name(name)+'/').decode()
    out=[]
    for anchor in re.finditer(r'<a ([^>]*)>',page):
        attrs=dict(re.findall(r'([\w-]+)="([^"]*)"',anchor.group(1)))
        url,_,digest=html.unescape(attrs.get('href','')).partition('#sha256=')
        filename=urllib.request.unquote(url.rsplit('/',1)[-1])
        if not filename.endswith('.whl'):continue
        if str(parse_wheel_filename(filename)[1])!=version:continue
        out.append(dict(filename=filename,url=url,sha256=digest,bytes=None,
                        metadata=url+'.metadata' if attrs.get('data-core-metadata') else None))
    return out


def pypi_release(name,version):
    data=json.loads(fetch(f'https://pypi.org/pypi/{canonicalize_name(name)}/{version}/json'))
    files=[dict(filename=u['filename'],url=u['url'],sha256=u['digests']['sha256'],bytes=u['size'],metadata=None)
           for u in data['urls'] if u['packagetype']=='bdist_wheel' and not u.get('yanked')]
    return files,data['info'].get('requires_dist') or []


def choose(files,python,name,version):
    ranks=supported_tags(python);best=None
    for item in files:
        rank=min((ranks[t] for t in parse_wheel_filename(item['filename'])[3] if t in ranks),default=None)
        if rank is not None and (best is None or rank<best[0]):best=(rank,item)
    if best is None:raise LookupError(f'No Linux x86_64 wheel for {name}=={version} (Python {python}).')
    return best[1]


def resolve(name,version,python):
    """Return (wheel entry, requires_dist) for one exact pin."""
    if '+' in version:
        item=choose(torch_candidates(name,version),python,name,version)
        if item['bytes'] is None:
            request=urllib.request.Request(item['url'],method='HEAD',headers={'User-Agent':'rgbmocap-linux-lock'})
            with urllib.request.urlopen(request,timeout=60) as response:item['bytes']=int(response.headers['Content-Length'])
        requires=[]
        if item['metadata']:
            text=fetch(item['metadata']).decode('utf-8','replace')
            requires=[line.split(':',1)[1].strip() for line in text.splitlines() if line.startswith('Requires-Dist:')]
        return item,requires
    files,requires=pypi_release(name,version)
    return choose(files,python,name,version),requires


def linux_only_requirements(requires,python):
    """Requirements that apply on Linux but not on Windows (never extras)."""
    linux,windows=environment(python,'linux'),environment(python,'windows')
    for text in requires:
        requirement=Requirement(text)
        marker=requirement.marker or Marker('python_version >= "0"')
        if marker.evaluate(linux) and not marker.evaluate(windows):yield requirement


def pinned(requirement,python):
    exact=[s.version for s in requirement.specifier if s.operator=='==' and '*' not in s.version]
    if exact:return exact[0]
    data=json.loads(fetch(f'https://pypi.org/pypi/{canonicalize_name(requirement.name)}/json'))
    versions=[v for v in data['releases'] if data['releases'][v] and requirement.specifier.contains(v,prereleases=False)]
    if not versions:raise LookupError(f'No release satisfies {requirement}')
    from packaging.version import Version
    return str(max(versions,key=Version))


def entry(package,item,**extra):
    return dict(name=package['name'],version=package['version'],filename=item['filename'],
                url=item['url'],sha256=item['sha256'],bytes=item['bytes'],**extra)


def convert_group(group,base_names):
    python=group['python'];packages=[];requires={}
    for package in group['packages']:
        if package.get('bundled'):
            packages.append(dict(package));continue
        if package['filename'].endswith('-none-any.whl'):
            packages.append(dict(package));requires[package['name']]=pypi_release(package['name'],package['version'])[1]
            continue
        item,needs=resolve(package['name'],package['version'],python)
        packages.append(entry(package,item));requires[package['name']]=needs
        print(f"  {group['key']}: {package['name']}=={package['version']} -> {item['filename']}",flush=True)
    present={canonicalize_name(p['name']) for p in packages}|base_names
    queue=[r for needs in requires.values() for r in linux_only_requirements(needs,python)]
    while queue:
        requirement=queue.pop(0);key=canonicalize_name(requirement.name)
        if key in present:continue
        version=pinned(requirement,python)
        item,needs=resolve(requirement.name,version,python)
        packages.append(entry(dict(name=key,version=version),item,linux_only=True));present.add(key)
        print(f"  {group['key']}: + {key}=={version} (Linux-only) -> {item['filename']}",flush=True)
        # Linux-only packages: every applicable dependency must be present, not only Linux-marked ones.
        for text in needs:
            dependency=Requirement(text)
            if dependency.marker is None or dependency.marker.evaluate(environment(python,'linux')):queue.append(dependency)
    out={k:v for k,v in group.items() if k!='packages'};out['packages']=packages
    return out


def uv_entry():
    files,_=pypi_release('uv',UV_VERSION)
    item=choose(files,'3.10.22','uv',UV_VERSION)
    return dict(version=UV_VERSION,filename=item['filename'],url=item['url'],sha256=item['sha256'],bytes=item['bytes'])


def blender_entry():
    listing=fetch(BLENDER.rsplit('/',1)[0]+'/blender-4.5.0.sha256').decode()
    name=BLENDER.rsplit('/',1)[1]
    digest=next(line.split()[0] for line in listing.splitlines() if line.strip().endswith(name))
    request=urllib.request.Request(BLENDER,method='HEAD',headers={'User-Agent':'rgbmocap-linux-lock'})
    with urllib.request.urlopen(request,timeout=60) as response:size=int(response.headers['Content-Length'])
    return dict(version='4.5.0',url=BLENDER,sha256=digest,bytes=size,archive_root=name.removesuffix('.tar.xz'))


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check',action='store_true',help='Fail if the committed Linux lock is stale.')
    args=parser.parse_args()
    raw=SOURCE.read_bytes();source=json.loads(raw)
    if args.check:
        current=json.loads(TARGET.read_text(encoding='utf8'))
        if current.get('source_lock_sha256')!=hashlib.sha256(raw).hexdigest():
            raise SystemExit('installer/runtime-lock-linux.json is stale: rerun tools/linux_lock.py.')
        print('Linux lock matches the Windows pins.');return
    groups=[];names={}
    for group in source['groups']:
        base=names.get(group.get('base'),set())
        converted=convert_group(group,base)
        names[group['key']]=base|{canonicalize_name(p['name']) for p in converted['packages']}
        groups.append(converted)
    lock=dict(schema=1,platform='linux-x86_64',glibc_min='.'.join(map(str,GLIBC)),
              source_lock_sha256=hashlib.sha256(raw).hexdigest(),source_lock_id=source.get('id'),
              registry_sha256=source['registry_sha256'],python_versions=source['python_versions'],
              uv=uv_entry(),blender=blender_entry(),groups=groups)
    lock['download_bytes']=sum(p['bytes'] or 0 for g in groups for p in g['packages'] if not p.get('bundled'))
    TARGET.write_text(json.dumps(lock,indent=1)+'\n',encoding='utf8')
    print('Wrote',TARGET.relative_to(ROOT),f"({lock['download_bytes']/2**30:.2f} GiB of wheels)")


if __name__=='__main__':
    sys.exit(main())
