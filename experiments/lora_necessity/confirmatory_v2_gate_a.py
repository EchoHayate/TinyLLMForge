import base64
import copy
import hashlib
import json
import os
from os import read as _io_read
from os import write as _io_write
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import time

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    build_confirmatory_tokenizer_projection,
    build_confirmatory_validation_batches,
    load_confirmatory_dataset,
    score_confirmatory_validation_batches,
)
from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import _plain
from recalibration_diagnostic_controller import (
    FALSE_FLAGS,
    SSH_STDERR_LIMIT,
    _check_hex,
    _claim_local_run_root,
    _cleanup_ssh,
    _pretty_json,
    _safe_raw_ns,
    _validate_preflight,
    _write_exclusive,
)
from recalibration_diagnostic_remote import QWEN3_8B_SHARD_SHA256
from recalibration_execution_package import BACKEND_SHA256
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json


TOKENIZER_SOURCE_PINS = {
    "pilot_protocol.py": "5074980a2b5b9d8ec0a52288d7cb62cbc847688f094cd8ba189a7135ba315c97",
    "validation_contract.py": "a4d6de138fa4817b23fc6e28442785b0d68a6b8f7360fb110582752d3463299d",
    "tokenizer_measurement.py": "527f63cc21fb9d044b179f0e67228944db36a9b279ef475208278adcdf73481e",
}
TOK_FILES_PINS = {
    key_name: QWEN3_8B_SHARD_SHA256[key_name]
    for key_name in ("config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
}
MULTI_BUNDLE_STDOUT_LIMIT = 16_777_216


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def build_multi_bundle_ssh_argv(ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime):
    backend_b64 = base64.b64encode(backend_raw).decode("ascii")
    expected_rt_b64 = base64.b64encode(encoded(expected_runtime)).decode("ascii")
    child_snippet = (
        "import json, pathlib\n"
        "expected_rt = json.loads(pathlib.Path('expected_runtime.json').read_text('utf-8'))\n"
        "backend_bytes = pathlib.Path('backend.py').read_bytes()\n"
        "namespace = {'__name__': '__main__', 'EXPECTED_RUNTIME': expected_rt}\n"
        "exec(compile(backend_bytes, 'validation_remote.py', 'exec'), namespace)\n"
    )
    multi_program = (
        "import base64,hashlib,json,os,shutil,signal,socket,subprocess,sys,tempfile,time\n"
        f"backend_bytes=base64.b64decode({backend_b64!r},validate=True)\n"
        "expected_rt_raw=base64.b64decode(" + repr(expected_rt_b64) + ",validate=True)\n"
        "expected_rt=json.loads(expected_rt_raw.decode('ascii'))\n"
        f"if hashlib.sha256(backend_bytes).hexdigest()!={BACKEND_SHA256!r}:\n"
        "    sys.exit(78)\n"
        "if socket.gethostname().split('.')[0]!=expected_rt['hostname'].split('.')[0] or os.getuid()!=expected_rt['uid']:\n"
        "    sys.exit(79)\n"
        "envelope=json.loads(sys.stdin.buffer.read(8388608).decode('ascii'))\n"
        "if envelope.get('kind')!='LORA_MULTI_VALIDATION_BUNDLES':\n"
        "    sys.exit(77)\n"
        "bundles_b64=envelope['bundles_b64']\n"
        "if type(bundles_b64) is not list or not (1<=len(bundles_b64)<=64):\n"
        "    sys.exit(76)\n"
        "os.umask(0o077)\n"
        "results_out=[]\n"
        "for b_idx,b_b64 in enumerate(bundles_b64):\n"
        "    bundle_in=base64.b64decode(b_b64,validate=True)\n"
        "    wdir=tempfile.mkdtemp(prefix='tlf-mb-',dir='/tmp')\n"
        "    t_start=time.monotonic_ns()\n"
        "    reaped=False\n"
        "    rc_val=None\n"
        "    out_bytes=b''\n"
        "    err_bytes=b''\n"
        "    timed_out=False\n"
        "    try:\n"
        "        os.chmod(wdir,0o700)\n"
        "        for fn,fb in (('backend.py',backend_bytes),('expected_runtime.json',expected_rt_raw)):\n"
        "            fdesc=os.open(os.path.join(wdir,fn),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)\n"
        "            with os.fdopen(fdesc,'wb') as fstream:\n"
        "                os.fchmod(fstream.fileno(),0o600)\n"
        "                fstream.write(fb)\n"
        "                fstream.flush()\n"
        f"        proc=subprocess.Popen([{resolved_interpreter!r},'-I','-S','-B','-c',{child_snippet!r}],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=wdir,env={{'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','HOME':'/nonexistent'}},close_fds=True,pass_fds=(),start_new_session=True,shell=False)\n"
        "        try:\n"
        "            out_bytes,err_bytes=proc.communicate(input=bundle_in,timeout=95.0)\n"
        "        except subprocess.TimeoutExpired:\n"
        "            timed_out=True\n"
        "        finally:\n"
        "            if proc.poll() is None:\n"
        "                try:\n"
        "                    os.killpg(proc.pid,signal.SIGKILL)\n"
        "                except ProcessLookupError:\n"
        "                    pass\n"
        "            try:\n"
        "                rc_val=proc.wait(timeout=2.0)\n"
        "                reaped=True\n"
        "            except subprocess.TimeoutExpired:\n"
        "                reaped=False\n"
        "            for pobj in (proc.stdin,proc.stdout,proc.stderr):\n"
        "                if pobj is not None:\n"
        "                    try:\n"
        "                        pobj.close()\n"
        "                    except OSError:\n"
        "                        pass\n"
        "    finally:\n"
        "        os.chdir('/')\n"
        "        shutil.rmtree(wdir,ignore_errors=True)\n"
        "    elapsed_ns=max(0,time.monotonic_ns()-t_start)\n"
        "    st_lbl=('timeout' if timed_out else ('completed' if (reaped and rc_val==0 and not err_bytes) else 'program_error'))\n"
        "    results_out.append({'batch_index':b_idx,'transport':{'status':st_lbl,'returncode':rc_val,'stdout_hex':out_bytes.hex(),'stderr_hex':err_bytes.hex(),'elapsed_ns':elapsed_ns,'process_reaped':reaped,'stdin_complete':(not timed_out)}})\n"
        "    if st_lbl!='completed':\n"
        "        break\n"
        "payload_out=json.dumps({'kind':'LORA_MULTI_VALIDATION_RESULTS','results':results_out},sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')+b'\\n'\n"
        "sys.stdout.buffer.write(payload_out)\n"
        "sys.stdout.buffer.flush()\n"
    )
    remote_cmd = "exec " + shlex.join((
        "env", "-i", "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "HOME=/nonexistent",
        resolved_interpreter, "-I", "-S", "-B", "-c", multi_program,
    ))
    return [*ssh_argv_prefix, remote_cmd]


def run_single_ssh_json_call(
    ssh_argv,
    ssh_environment,
    stdin_raw,
    cancellations,
    secondary_errors,
    *,
    timeout_ns=1_800_000_000_000,
    stdout_cap=MULTI_BUNDLE_STDOUT_LIMIT,
):
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    stdin_offset = 0
    stdin_closed = False
    stdout_eof = False
    stderr_eof = False
    limit_status = None
    cleanup_info = None
    process = None
    selector = None
    constructor_completed = False
    started_ns = _safe_raw_ns() or 0

    def close_stdin():
        nonlocal stdin_closed
        if stdin_closed:
            return
        stdin_closed = True
        if selector is not None and process is not None and process.stdin is not None:
            try:
                selector.unregister(process.stdin)
            except (KeyError, ValueError):
                pass
        pipe_obj = getattr(process, "stdin", None)
        if pipe_obj is not None:
            try:
                pipe_obj.close()
            except BaseException as err:
                if not isinstance(err, Exception) and not cancellations:
                    cancellations.append((err, err.__traceback__))
                else:
                    secondary_errors.append({"phase": "mb_close_stdin", "type": type(err).__name__})

    try:
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + timeout_ns
        process = subprocess.Popen(
            ssh_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=dict(ssh_environment),
            close_fds=True,
            start_new_session=True,
            shell=False,
        )
        constructor_completed = True
        selector = selectors.DefaultSelector()
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        if stdin_raw:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            close_stdin()

        while True:
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                limit_status = "timeout"
                break
            if not selector.get_map() and process.poll() is not None:
                break
            rem_sec = (deadline_ns - now_ns) / 1_000_000_000.0
            if not selector.get_map():
                time.sleep(min(0.02, rem_sec))
                continue
            events = selector.select(min(0.05, rem_sec))
            if time.monotonic_ns() >= deadline_ns:
                limit_status = "timeout"
                break
            for key, _mask in events:
                if key.data == "stdin":
                    if stdin_closed:
                        continue
                    try:
                        written = _io_write(
                            key.fd, stdin_raw[stdin_offset:stdin_offset + 65536]
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        close_stdin()
                        continue
                    if written > 0:
                        stdin_offset += written
                    if stdin_offset >= len(stdin_raw):
                        close_stdin()
                    continue

                target_buf, cap = (
                    (stdout_buf, stdout_cap)
                    if key.data == "stdout"
                    else (stderr_buf, SSH_STDERR_LIMIT)
                )
                rem_bytes = cap - len(target_buf)
                try:
                    chunk = _io_read(key.fd, min(65536, rem_bytes + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except (KeyError, ValueError):
                        pass
                    if key.data == "stdout":
                        stdout_eof = True
                    else:
                        stderr_eof = True
                else:
                    if len(chunk) > rem_bytes:
                        target_buf.extend(chunk[:rem_bytes + 1])
                        limit_status = "output_limit"
                        break
                    target_buf.extend(chunk)
            if limit_status is not None:
                break
    except BaseException as run_err:
        if not isinstance(run_err, Exception) and not cancellations:
            cancellations.append((run_err, run_err.__traceback__))
        else:
            secondary_errors.append({"phase": "mb_ssh_collect", "type": type(run_err).__name__})
        limit_status = limit_status or "launch_failed"
    finally:
        child_pid = getattr(process, "pid", None)
        if type(child_pid) is int and child_pid > 1:
            cleanup_info = _cleanup_ssh(
                process, constructor_completed, cancellations, secondary_errors
            )
        for res_obj in (
            selector,
            None if stdin_closed else getattr(process, "stdin", None),
            getattr(process, "stdout", None),
            getattr(process, "stderr", None),
        ):
            if res_obj is not None:
                try:
                    res_obj.close()
                except BaseException as close_err:
                    if not isinstance(close_err, Exception) and not cancellations:
                        cancellations.append((close_err, close_err.__traceback__))
                    else:
                        secondary_errors.append({"phase": "mb_close_stream", "type": type(close_err).__name__})

    reaped = bool(cleanup_info and cleanup_info["child_reaped"])
    returncode = cleanup_info["returncode"] if reaped else None
    group_absent = bool(cleanup_info and cleanup_info["group_absent"])
    cleanup_bounded = bool(cleanup_info and cleanup_info["cleanup_bounded"])
    stdin_complete = stdin_offset == len(stdin_raw)
    status_label = (
        limit_status
        if limit_status is not None
        else ("completed" if (reaped and returncode == 0 and stdin_complete and stdout_eof and stderr_eof) else "program_error")
    )
    return {
        "status": status_label,
        "returncode": returncode,
        "stdout_bytes": bytes(stdout_buf),
        "stderr_bytes": bytes(stderr_buf),
        "reaped": reaped,
        "group_absent": group_absent,
        "cleanup_bounded": cleanup_bounded,
    }


def _build_tokenizer_ssh_argv(
    ssh_argv_prefix,
    interpreter,
    site_packages,
    checkpoint_root,
    expected_tokenizer_sha256,
    expected_hostname,
    expected_uid,
    tok_sources,
    projection_sha256,
    remote_parent,
    stage_id,
):
    sources_b64 = {
        fname: base64.b64encode(tok_sources[fname]).decode("ascii")
        for fname in sorted(TOKENIZER_SOURCE_PINS)
    }
    remote_cfg = {
        "hostname": expected_hostname.split(".")[0],
        "uid": expected_uid,
        "site_packages": site_packages,
        "checkpoint_root": checkpoint_root,
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        "tok_files_sha256": dict(TOK_FILES_PINS),
        "sources_sha256": dict(TOKENIZER_SOURCE_PINS),
        "projection_sha256": projection_sha256,
        "remote_parent": remote_parent,
        "stage_id": stage_id,
    }
    cfg_b64 = base64.b64encode(encoded(remote_cfg)).decode("ascii")
    src_map_b64 = base64.b64encode(encoded(sources_b64)).decode("ascii")
    tok_program = (
        "import base64,hashlib,json,os,pathlib,shutil,socket,sys\n"
        f"cfg=json.loads(base64.b64decode({cfg_b64!r},validate=True).decode('ascii'))\n"
        f"src_map=json.loads(base64.b64decode({src_map_b64!r},validate=True).decode('ascii'))\n"
        "if socket.gethostname().split('.')[0]!=cfg['hostname'] or os.getuid()!=cfg['uid']:\n"
        "    sys.exit(79)\n"
        "proj_raw=sys.stdin.buffer.read(8388608)\n"
        "if hashlib.sha256(proj_raw).hexdigest()!=cfg['projection_sha256']:\n"
        "    sys.exit(78)\n"
        "chk_root=pathlib.Path(cfg['checkpoint_root']).resolve(strict=True)\n"
        "for rel_n,exp_sha in cfg['tok_files_sha256'].items():\n"
        "    fbytes=(chk_root/rel_n).read_bytes()\n"
        "    if hashlib.sha256(fbytes).hexdigest()!=exp_sha:\n"
        "        sys.exit(77)\n"
        "os.umask(0o077)\n"
        "pdir=pathlib.Path(cfg['remote_parent'])\n"
        "pdir.mkdir(mode=0o700,exist_ok=True)\n"
        "wdir=pdir/('tok-g4a-'+cfg['stage_id'])\n"
        "wdir.mkdir(mode=0o700)\n"
        "try:\n"
        "    for fn,b64_txt in src_map.items():\n"
        "        raw_mod=base64.b64decode(b64_txt,validate=True)\n"
        "        if hashlib.sha256(raw_mod).hexdigest()!=cfg['sources_sha256'][fn]:\n"
        "            sys.exit(76)\n"
        "        (wdir/fn).write_bytes(raw_mod)\n"
        "    os.environ.update({'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','USE_TORCH':'0','USE_TF':'0','USE_FLAX':'0','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','HOME':str(wdir),'TMPDIR':str(wdir),'HF_HOME':str(wdir/'hf-cache')})\n"
        "    sys.path.extend([str(wdir),cfg['site_packages']])\n"
        "    import transformers\n"
        "    from tokenizer_measurement import _state,measure\n"
        "    from validation_contract import digest,encoded,parse_json\n"
        "    tok=transformers.AutoTokenizer.from_pretrained(str(chk_root),use_fast=True,local_files_only=True,trust_remote_code=False)\n"
        "    init_sha=digest(_state(tok)[0])\n"
        "    if init_sha!=cfg['expected_tokenizer_sha256']:\n"
        "        sys.exit(75)\n"
        "    proj_obj=parse_json(proj_raw,limit=8388608)\n"
        "    out_rows=[]\n"
        "    for row in proj_obj['rows']:\n"
        "        mres=measure(tok,row['messages'],row['answer'])\n"
        "        out_rows.append({'ordinal':row['ordinal'],'status':mres['status'],'state_sha256':mres['state_sha256'],'input_tokens':mres['input']['input_tokens'],'target_tokens':(mres['target']['target_tokens'] if mres['target'] is not None else None)})\n"
        "    fin_sha=digest(_state(tok)[0])\n"
        "    if fin_sha!=init_sha:\n"
        "        sys.exit(74)\n"
        "    sys.stdout.buffer.write(encoded({'kind':'LORA_CONFIRMATORY_V2_TOK_RESULT','tokenizer_state_sha256':fin_sha,'records':out_rows})+b'\\n')\n"
        "    sys.stdout.buffer.flush()\n"
        "finally:\n"
        "    shutil.rmtree(wdir,ignore_errors=True)\n"
    )
    remote_cmd = "exec " + shlex.join((
        "env", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH", "-u", "PYTHONHOME",
        "-u", "PYTHONPATH", interpreter, "-I", "-S", "-B", "-c", tok_program,
    ))
    return [*ssh_argv_prefix, remote_cmd]


def build_gate_a_package(
    raw_files,
    worker_raw,
    backend_raw,
    tok_sources,
    preflight_raw,
    *,
    stage_id,
    dynamic_run_ids,
    ssh_argv_prefix,
    ssh_environment,
    interpreter,
    resolved_interpreter,
    site_packages,
    checkpoint_root,
    expected_tokenizer_sha256,
    remote_parent,
    local_run_root,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    load_confirmatory_dataset(raw_files)
    require(type(worker_raw) is bytes and _sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(type(backend_raw) is bytes and _sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    fields(tok_sources, TOKENIZER_SOURCE_PINS)
    for fname, exp_sha in TOKENIZER_SOURCE_PINS.items():
        require(_sha256(tok_sources[fname]) == exp_sha, "tok source pin: " + fname)
    _check_hex(stage_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    batches = build_confirmatory_validation_batches(raw_files, worker_raw, dynamic_run_ids)
    tok_proj = build_confirmatory_tokenizer_projection(raw_files)
    tok_proj_raw = encoded(tok_proj)
    tok_proj_sha = _sha256(tok_proj_raw)

    dynamic_ssh_argv = build_multi_bundle_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )
    tok_ssh_argv = _build_tokenizer_ssh_argv(
        ssh_argv_prefix,
        interpreter,
        site_packages,
        checkpoint_root,
        expected_tokenizer_sha256,
        expected_runtime["hostname"],
        expected_runtime["uid"],
        tok_sources,
        tok_proj_sha,
        remote_parent,
        stage_id,
    )

    manifest = {
        "kind": "LORA_CONFIRMATORY_V2_GATE_A_PACKAGE",
        "version": 2,
        "data_version": "confirmatory_v2_r1",
        "stage_id": stage_id,
        "dynamic_run_ids": list(dynamic_run_ids),
        "file_sha256": dict(CONFIRMATORY_FILE_PINS),
        "preflight_receipt_sha256": _sha256(preflight_raw),
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        "tokenizer_projection_sha256": tok_proj_sha,
        "dynamic_argv_sha256": digest(dynamic_ssh_argv),
        "tokenizer_argv_sha256": digest(tok_ssh_argv),
        "local_run_root": local_run_root,
        "planned_dynamic_batches": len(batches),
        "planned_tokenizer_rows": len(tok_proj["rows"]),
        "single_ssh_multi_bundle": True,
        **{flag: False for flag in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "dynamic_ssh_argv": dynamic_ssh_argv,
            "tokenizer_ssh_argv": tok_ssh_argv,
            "ssh_environment": dict(ssh_environment),
            "tokenizer_projection_hex": tok_proj_raw.hex(),
            "expected_runtime": copy.deepcopy(expected_runtime),
        },
    }


def summarize_tokenizer_audit_records(raw_files, records, *, expected_tokenizer_sha256):
    proj = build_confirmatory_tokenizer_projection(raw_files)
    require(type(records) is list and len(records) == len(proj["rows"]), "200 records required")
    by_category = {}
    in_tokens = []
    tg_tokens = []
    all_ok = True

    for row_meta, rec in zip(proj["rows"], records):
        require(rec["ordinal"] == row_meta["ordinal"], "ordinal mismatch")
        require(rec["state_sha256"] == expected_tokenizer_sha256, "state_sha256 mismatch")
        cat = row_meta["category"]
        cat_stat = by_category.setdefault(cat, {"scheduled": 0, "checked": 0, "over_limit": 0, "max_input_tokens": 0})
        cat_stat["scheduled"] += 1
        cat_stat["checked"] += 1
        in_val = rec["input_tokens"]
        tg_val = rec["target_tokens"]
        in_tokens.append(in_val)
        cat_stat["max_input_tokens"] = max(cat_stat["max_input_tokens"], in_val)
        if row_meta["answer"] is not None:
            if type(tg_val) is int and tg_val > 0:
                tg_tokens.append(tg_val)
            else:
                all_ok = False
        else:
            require(tg_val is None, "unexpected target_tokens")
        within = (
            rec["status"] == "OK"
            and type(in_val) is int
            and 0 < in_val <= 4096
            and (tg_val is None or 0 < tg_val <= 256)
        )
        if not within:
            cat_stat["over_limit"] += 1
            all_ok = False

    return {
        "kind": "LORA_CONFIRMATORY_V2_TOKENIZER_SUMMARY",
        "version": 1,
        "checked_rows": len(records),
        "all_200_rows_within_budget": all_ok,
        "max_input_tokens": max(in_tokens),
        "min_input_tokens": min(in_tokens),
        "max_target_tokens": max(tg_tokens),
        "min_target_tokens": min(tg_tokens),
        "by_category": by_category,
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        **{flag: False for flag in FLAGS},
    }


def run_gate_a_stage(package, raw_files, worker_raw, backend_raw):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]
    manifest_no_sha = {key_n: val_v for key_n, val_v in pub.items() if key_n != "package_sha256"}
    require(digest(manifest_no_sha) == pub["package_sha256"], "package_sha256 mismatch")
    require(_sha256(backend_raw) == BACKEND_SHA256, "backend_raw pin")
    require(_sha256(worker_raw) == WORKER_SHA256, "worker_raw pin")
    require(digest(priv["dynamic_ssh_argv"]) == pub["dynamic_argv_sha256"], "dynamic argv sha")
    require(digest(priv["tokenizer_ssh_argv"]) == pub["tokenizer_argv_sha256"], "tokenizer argv sha")

    batches = build_confirmatory_validation_batches(
        raw_files, worker_raw, pub["dynamic_run_ids"]
    )
    stage_id = pub["stage_id"]
    run_fd = _claim_local_run_root(pub["local_run_root"], stage_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    stage_error = None
    batch_records = []
    dynamic_report = None
    tok_summary = None
    all_clean = True

    def safe_write(fname, fbytes):
        try:
            _write_exclusive(run_fd, fname, fbytes)
        except BaseException as w_err:
            persistence_errors.append(w_err)
            if not isinstance(w_err, Exception) and not cancellations:
                cancellations.append((w_err, w_err.__traceback__))

    try:
        safe_write("gate-a-package-public.json", _pretty_json(pub))
        try:
            multi_stdin = encoded({
                "kind": "LORA_MULTI_VALIDATION_BUNDLES",
                "bundles_b64": [
                    base64.b64encode(b_meta["bundle_raw"]).decode("ascii")
                    for b_meta in batches
                ],
            })
            dyn_call = run_single_ssh_json_call(
                priv["dynamic_ssh_argv"],
                priv["ssh_environment"],
                multi_stdin,
                cancellations,
                secondary_errors,
                timeout_ns=1_200_000_000_000,
            )
            if not (dyn_call["reaped"] and dyn_call["group_absent"] and dyn_call["cleanup_bounded"]):
                all_clean = False
            safe_write("dynamic-multi-stdout.json", dyn_call["stdout_bytes"])
            safe_write("dynamic-multi-stderr.bin", dyn_call["stderr_bytes"])
            require(dyn_call["status"] == "completed", "dynamic multi-bundle SSH failed")
            dyn_envelope = parse_json(dyn_call["stdout_bytes"].strip(), limit=MULTI_BUNDLE_STDOUT_LIMIT)
            require(dyn_envelope.get("kind") == "LORA_MULTI_VALIDATION_RESULTS", "multi result kind")
            batch_records = dyn_envelope["results"]
            for b_meta, b_rec in zip(batches, batch_records):
                b_idx = b_meta["batch_index"]
                safe_write("dyn-batch-%02d-bundle.json" % b_idx, b_meta["bundle_raw"])
                safe_write(
                    "dyn-batch-%02d-stdout.jsonl" % b_idx,
                    bytes.fromhex(b_rec["transport"]["stdout_hex"]),
                )
            dynamic_report = score_confirmatory_validation_batches(
                raw_files,
                worker_raw,
                pub["dynamic_run_ids"],
                batch_records,
                priv["expected_runtime"],
            )
            safe_write("dynamic-validation-report.json", _pretty_json(dynamic_report))

            time.sleep(3.0)

            tok_proj_raw = bytes.fromhex(priv["tokenizer_projection_hex"])
            require(_sha256(tok_proj_raw) == pub["tokenizer_projection_sha256"], "projection sha")
            tok_res = run_single_ssh_json_call(
                priv["tokenizer_ssh_argv"],
                priv["ssh_environment"],
                tok_proj_raw,
                cancellations,
                secondary_errors,
                timeout_ns=900_000_000_000,
            )
            if not (tok_res["reaped"] and tok_res["group_absent"] and tok_res["cleanup_bounded"]):
                all_clean = False
            safe_write("tokenizer-stdout.json", tok_res["stdout_bytes"])
            safe_write("tokenizer-stderr.bin", tok_res["stderr_bytes"])
            require(tok_res["status"] == "completed", "tokenizer SSH failed")
            tok_payload = parse_json(tok_res["stdout_bytes"].strip(), limit=8388608)
            require(
                tok_payload.get("kind") == "LORA_CONFIRMATORY_V2_TOK_RESULT"
                and tok_payload.get("tokenizer_state_sha256") == pub["expected_tokenizer_sha256"],
                "tokenizer payload identity",
            )
            tok_summary = summarize_tokenizer_audit_records(
                raw_files,
                tok_payload["records"],
                expected_tokenizer_sha256=pub["expected_tokenizer_sha256"],
            )
            safe_write("tokenizer-audit-summary.json", _pretty_json(tok_summary))
        except BaseException as body_err:
            if not isinstance(body_err, Exception) and not cancellations:
                cancellations.append((body_err, body_err.__traceback__))
            else:
                stage_error = {"type": type(body_err).__name__, "message": str(body_err)[:512]}

        summary = {
            "kind": "LORA_CONFIRMATORY_V2_GATE_A_EVIDENCE",
            "version": 2,
            "stage_id": stage_id,
            "package_sha256": pub["package_sha256"],
            "dynamic_batches_completed": len(batch_records),
            "dynamic_validation_passed": bool(dynamic_report and dynamic_report["all_checks_passed"] and stage_error is None),
            "tokenizer_audit_passed": bool(tok_summary and tok_summary["all_200_rows_within_budget"] and stage_error is None),
            "tokenizer_summary": tok_summary,
            "stage_error": stage_error,
            "secondary_errors": secondary_errors,
            "binding_complete": bool(all_clean and stage_error is None and not cancellations and not persistence_errors and not secondary_errors),
            **{flag: False for flag in FALSE_FLAGS},
        }
        safe_write("gate-a-controller-summary.json", _pretty_json(summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return summary
