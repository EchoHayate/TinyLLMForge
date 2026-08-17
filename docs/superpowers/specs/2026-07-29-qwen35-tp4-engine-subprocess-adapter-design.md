# Qwen3.5 TP4 Engine Subprocess Adapter Design

## Goal

Add the missing local process boundary that can execute an independently
verified and single-use-authorized TP4 Engine remote execution plan without
weakening the process-free executor core.

## Architecture

Keep `qwen35_tp4_engine_remote_execution_executor.py` free of subprocess
imports and default runners. Add one isolated adapter module that implements
the existing injected `command_runner(name, argv, stdout_path, env)` protocol.
The adapter has no CLI or import-time execution and is never invoked by tests
with real SSH.

The adapter accepts only:

- `ssh`;
- `scp`;
- the exact current Python executable used by local plan commands.

It always uses `shell=False`, rejects malformed argv and unexpected
environment keys, and merges only the exact required Kerberos cache binding
into a copied local environment.

## Output Handling

Normal commands write stdout and stderr to temporary regular files. After the
child exits, the adapter reads at most the configured log limit plus one byte
and rejects oversized or non-UTF-8 logs.

`package_download` writes stdout directly to the requested new regular file in
binary mode. It captures only stderr as bounded UTF-8 text and returns an
independently computed output SHA-256 and byte size. Existing output paths,
symlinks, empty package output, and failed commands never produce a valid
package identity.

## Safety Boundary

The adapter does not:

- verify plans;
- consume authorization;
- classify receipts;
- expose a `main` or `__main__` entrypoint;
- select GPUs;
- create remote commands;
- retry or kill unrelated processes.

Those responsibilities remain in the existing plan, authorization, executor,
receipt, and remote resource guard modules.

## Testing

CPU-only tests inject a fake `Popen`-compatible factory and prove:

- exact argv, `shell=False`, and environment propagation;
- executable allowlisting;
- bounded UTF-8 log handling;
- binary package streaming and SHA/size reporting;
- rejection of existing package destinations;
- no CLI or default execution;
- compatibility with the existing offline executor protocol.

No test opens SSH, executes `scp`, calls `nvidia-smi`, imports Torch, or starts
a GPU workload.
