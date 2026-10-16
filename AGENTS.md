# AGENTS.md: instructions for an AI coding agent

You are setting up and running a GPU benchmark on this PC for its owner. Follow these steps in
order, run the commands exactly as written, and report back. The whole job takes about 45-75
minutes, most of it unattended (setup 5-25 min, benchmark 35-50 min).

**Rules**

- Do **not** edit any code, configuration or settings in this repository to make a scenario pass.
  Do not change batch sizes, timeouts or the scenario list. `oom` (out of memory) and `skipped` rows
  are valid results.
- Do not add options to the run commands below. The only exceptions: `--gpu N` when step 1 tells
  you to, and `--retry-oom` when the troubleshooting table tells you to. In particular `--tiny`,
  `--smoke`, `--device cpu`, `--mem-cap-gib`, `--allow-cpu-only` and `--no-resume` are test options
  for the maintainers: their numbers are not benchmark results.
- Run every command from the repository root (the folder that contains this file).
- Do not upload the results anywhere. The only network use is the download of PyTorch and the public
  model weights.
- Never print or store a Hugging Face token in any file.
- If something fails that the troubleshooting table below does not cover, stop and report the exact
  error text and the last 40 lines of the relevant log to the human.

**If your tool starts a new shell for every command** (most agent tools do), environment changes do
not carry over from one command to the next. Put the PATH line in front of every `uv` command in the
same command line, for example
`export PATH="$HOME/.local/bin:$PATH"; uv run --no-sync python -m agbench doctor` (bash) or
`$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"; uv run --no-sync python -m agbench doctor`
(PowerShell), or call uv by its full path (`~/.local/bin/uv`, `& "$env:USERPROFILE\.local\bin\uv.exe"`).
Environment variables such as `HF_ENDPOINT` must also be set in the same command line as the command
that needs them.

## 1. Pre-flight checks

Run `nvidia-smi` and read the **driver version** and GPU name:

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
```

If more than one GPU is listed, note the `index` of the RTX 5060 Ti (also shown by `nvidia-smi -L`)
and add `--gpu N` (that index) to the `doctor`, `quick` and `full` commands below.

| Driver version | What to do |
|---|---|
| 580 or newer | use `--extra cu130` (PyTorch 2.14.1 for CUDA 13.0); this is the normal case |
| 570 to 579 | use `--extra cu128` (PyTorch 2.11.0 for CUDA 12.8); or ask the human to update the driver |
| below 570, or `nvidia-smi` not found | stop and ask the human to install a current NVIDIA driver, then reboot. **Windows:** the GeForce driver from nvidia.com or the NVIDIA App. **Linux (Ubuntu/Debian):** `sudo ubuntu-drivers install`, or `sudo apt install nvidia-driver-580-open`; the package name must end in **`-open`**, because RTX 50 cards only work with NVIDIA's open kernel modules (the package without `-open` does not support them). **Fedora:** `akmod-nvidia-open` from RPM Fusion. With Secure Boot on, the human must enroll the MOK key on the blue screen during the reboot. Check afterwards: `cat /proc/driver/nvidia/version` must mention `Open Kernel Module`, and `nvidia-smi` must list the GPU. |

Check free disk space. About **25 GB** are needed: Python environment 5-7 GB, uv download cache of
similar size, models 3.3 GB, results < 50 MB.

```bash
df -h .                                   # Linux, and Git Bash on Windows
```
```powershell
Get-PSDrive -PSProvider FileSystem        # Windows PowerShell: look at the Free column of the drive
```

Also check that the PC is plugged in (laptops: on AC power, "best performance" power mode). The
benchmark itself asks the operating system not to go to sleep while it runs.

`git` is only needed if you still have to download the repository (`git --version`). On Windows,
`winget install --id Git.Git -e` installs it; afterwards open a new shell, or call it as
`"C:\Program Files\Git\cmd\git.exe"`.

## 2. Install uv, get the code, install the environment

**If the current folder already contains `AGENTS.md` and `pyproject.toml`, you are already in the
repository:** skip `git clone` and `cd` below; only install uv and run `uv sync`. Otherwise clone it
first.

**Linux (bash):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/Kaiser0207/antispoof-gpu-bench.git     # skip if already inside it
cd antispoof-gpu-bench                                               # skip if already inside it
uv sync --frozen --extra cu130          # or --extra cu128 for driver 570-579
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
git clone https://github.com/Kaiser0207/antispoof-gpu-bench.git     # skip if already inside it
cd antispoof-gpu-bench                                               # skip if already inside it
uv sync --frozen --extra cu130          # or --extra cu128 for driver 570-579
```

**Windows from Git Bash** (for example Claude Code on Windows, which runs commands in Git Bash). Do
not use the Linux installer there; install the Windows uv:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
export PATH="$HOME/.local/bin:$PATH"      # HOME is /c/Users/<name>
uv sync --frozen --extra cu130            # or --extra cu128 for driver 570-579
```

In Git Bash use the bash forms of the commands below (`df -h .`, `export NAME=value`), and
`powershell -ExecutionPolicy ByPass -File scripts/bench.ps1` instead of `scripts/bench.sh`.

`uv sync` downloads Python 3.12 if needed and PyTorch with its CUDA libraries: about **4-5 GB** on
Linux and about 3 GB on Windows. Allow 2-15 minutes. `--frozen` installs exactly the versions in
`uv.lock`; never run `uv lock` or `uv add`.

From now on **always** run commands as `uv run --no-sync python -m agbench ...`. A plain `uv run`
re-syncs the environment without the extra and removes PyTorch.

(Shortcut: `bash scripts/bench.sh` or `powershell -ExecutionPolicy ByPass -File scripts\bench.ps1`
does steps 2-6 in one go: it picks cu130/cu128 from the driver version and runs the `full` suite.
It runs in the foreground, so it has the same time-limit problem as step 6. The step-by-step route
below is easier to debug.)

## 3. Check the environment

```bash
uv run --no-sync python -m agbench doctor
```

The last line must be `DOCTOR: OK` or `DOCTOR: WARN (n)`. Warnings are fine (read them: for example,
another program is holding GPU memory). `DOCTOR: FATAL (n)`: fix the failed checks with the
troubleshooting table before going on; each FATAL line says what to do. The doctor also writes
`results/doctor_<gpu>_<date>.json`.

## 4. Download the models (3.3 GB, once)

```bash
uv run --no-sync python -m agbench fetch
```

In the table printed at the end, the `status` column must say `cached` or `downloaded` for
`w2v2b`, `xlsr`, `w2v2l` and `wavlmb`. A warning about unauthenticated requests or `HF_TOKEN` is
harmless; do not ask for a token unless you get HTTP 429 (see troubleshooting). If a download
fails, run the same command again (it resumes). Behind a firewall or with a slow connection see
"Hugging Face" in the troubleshooting table.

## 5. Prepare the PC

- Close games, browsers with video, other GPU programs, and anything heavy on the CPU.
- Windows only, optional: in **NVIDIA Control Panel > Manage 3D settings > Global Settings**, set
  **"CUDA - Sysmem Fallback Policy"** to **"Prefer No Sysmem Fallback"** (it can be set back
  afterwards). This stops the driver from silently using system RAM when GPU memory is full. The
  benchmark also guards against this itself, so skip this if the setting is not there.
- Do not use the PC during the run; it measures speed.

## 6. Run the benchmark: `quick`, then `full`

Run `quick` (about **10-15 minutes**), and when it has finished, run `full` straight away (about
**20-35 minutes** more). `full` continues where `quick` stopped and repeats nothing. Skip `full` only
if the human has already said they cannot spare the time.

**These commands take longer than most agent tool time limits (often 2-10 minutes).** Start each one
in the background with its output going to a file, then read that file every 1-2 minutes until the
lines `DONE ...` and `Report: ...` appear. Do not stop a run that is still printing. If your tool has
its own way to run a command in the background, use it with the commands below. Otherwise:

Linux (bash):

```bash
setsid nohup uv run --no-sync python -m agbench quick > results/console_quick.log 2>&1 < /dev/null &
```

Windows (PowerShell; from Git Bash wrap it as `powershell -NoProfile -Command "..."`):

```powershell
Start-Process -WindowStyle Hidden -FilePath uv -ArgumentList 'run','--no-sync','python','-m','agbench','quick' -RedirectStandardOutput results\console_quick.log -RedirectStandardError results\console_quick.err
```

(With a new shell per command, put the PATH line in front, as shown at the top.) For `full`, use the
same command with `full` instead of `quick` and `console_full.log` / `console_full.err`. Check the
progress with `tail -n 5 results/console_quick.log` (bash) or
`Get-Content results\console_quick.log -Tail 5` (PowerShell).

One line per scenario is printed, for example
`[ 7/18] train.xlsr.6s.bs16.bf16 ... ok (8x2 after OOM at 16x1) 19.6 clips/s, 9.4 GiB, 61 s [elapsed ...]`.
Lines with `oom` or `skipped` are normal results. A single scenario can take up to 8 minutes without
printing anything (the 5-minute sustained run in `full`).

If the run was stopped anyway (the log has had no new line for more than 10 minutes and has no
`DONE` line, or the PC slept or rebooted), run the **same command** again: it continues where it
stopped, and stops leftover benchmark processes from the killed run first.

## 7. Verify

- The log ends with `DONE in ...: ok N, oom N, error N, timeout N, crashed N, skipped N, cached N -> results/<gpu>_<date>.jsonl`
  followed by `Report: results/<gpu>_<date>_report/summary.md`.
- These exist: `results/<gpu>_<date>.jsonl` and `results/<gpu>_<date>_report/summary.md` (with
  `comparison.csv` and `figures/`).
- If `error`, `timeout` or `crashed` is not 0 in the `DONE` line, run the same command once more
  (failed scenarios are retried once); if a row fails again, it is still a valid report entry, so
  just send the results. (In the foreground the exit code says the same: 0 = every scenario ended
  `ok`, `oom` or `skipped`; 1 = at least one `error`, `timeout` or `crashed`; 2 = a setup problem,
  see step 3.)

## 8. Send back

Zip the whole `results/` folder (it contains `*.jsonl`, `doctor_*.json`, `*_report/summary.md` with
figures, the console logs and `logs/`); nothing else is needed:

```bash
uv run --no-sync python -m zipfile -c agbench_results.zip results/
```

Give `agbench_results.zip` to the human to send. Also tell the human the final `DONE ...` line of
each run and anything unusual you saw. The results contain no user name, host name, IP address,
e-mail address or names of other programs; home-directory paths are replaced by `~`.

## Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| doctor check 6 FATAL: driver older than needed, or `CUDA driver version is insufficient` | driver < 580 with the cu130 build | update the NVIDIA driver (step 1), or run the `uv sync --frozen --extra cu128` command that the message names if the driver is 570-579 |
| `nvidia-smi`: `No devices were found` on an RTX 50 card (Linux); dmesg says the GPU needs the open kernel modules | the proprietary driver package is installed | install the `-open` package (step 1) and reboot |
| `no kernel image is available for execution on the device`, or doctor check 4/5 (arch / kernel smoke) FATAL | wrong PyTorch build (CPU-only, or a CUDA 12.6 build without Blackwell `sm_120` kernels) | run the `uv sync --frozen --extra ...` command from step 2 again, then `uv run --no-sync python -m agbench doctor`; never install torch with pip by hand |
| `torch is not installed: run "uv sync ..."` | a plain `uv run` (without `--no-sync`) re-synced and removed torch | run the `uv sync` command again and always use `uv run --no-sync` |
| `The lockfile at uv.lock needs to be updated, but --locked was provided` | an old command line or a newer uv | use `uv sync --frozen --extra cu130` (or `cu128`); never run `uv lock` |
| `uv: command not found` / `'uv' is not recognized` | the shell did not keep the PATH change | see "If your tool starts a new shell for every command" at the top |
| `torch.cuda.is_available()` is False / doctor check 3 FATAL | driver not loaded, GPU not visible, CPU-only torch, or running inside a VM/container without GPU access | `nvidia-smi` must work first; reboot after driver installs; in WSL2 install the Windows driver only (not a Linux driver) |
| rows with status `oom` | the scenario does not fit in this GPU's memory even at the smallest split | expected on some rows (especially 8 GB cards); it is data, do not change settings |
| many rows `oom` that should fit, doctor warned about used VRAM | another program held GPU memory | close it (check `nvidia-smi`), then run the same command again with `--retry-oom` added; tell the human which rows were affected |
| Windows: `An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | a bug in the benchmark's worker start-up | do not patch it: report it with the log file |
| Windows: the first run is very slow to start | Windows Defender scanning the new Python environment | wait, or ask the human to add the repository folder as a Defender exclusion |
| `No space left on device` / disk full | caches | free space (about 25 GB needed); uv's cache can be cleared with `uv cache clean` after setup |
| Hugging Face HTTP 429 (rate limit) or timeouts during `fetch` | network / anonymous rate limit | wait a few minutes and rerun `fetch`; or ask the human for a free read token and set `HF_TOKEN` in the same command line as `fetch` (never write it to a file); slow network: `HF_HUB_DOWNLOAD_TIMEOUT=60` |
| huggingface.co blocked | firewall / region | set `HF_ENDPOINT` to a Hugging Face mirror and `HF_HUB_DISABLE_XET=1` in the same command line as `fetch`, e.g. `HF_ENDPOINT=https://<mirror> HF_HUB_DISABLE_XET=1 uv run --no-sync python -m agbench fetch` (PowerShell: `$env:HF_ENDPOINT='https://<mirror>'; $env:HF_HUB_DISABLE_XET='1'; uv run --no-sync python -m agbench fetch`). Downloads are checked against the pinned file size and SHA-256 |
| no internet on this PC | offline | on another PC run `fetch`, copy its Hugging Face cache folder (`~/.cache/huggingface/hub`, Windows `%USERPROFILE%\.cache\huggingface\hub`) to the same place here and set `HF_HUB_OFFLINE=1`; or put the files as `<dir>/<key>/config.json` and `<dir>/<key>/pytorch_model.bin` (keys `w2v2b`, `xlsr`, `w2v2l`, `wavlmb`) and set `AGBENCH_MODELS_DIR=<dir>` |
| `ERROR: another agbench run (pid ...)` | a previous run is still running (lock file `results/.lock`) | wait for it (check its console log); a lock left by a dead process is removed automatically |
| the run stops every few minutes | the agent tool's time limit kills the command | start it in the background (step 6); each rerun resumes |
| WSL2 | works, but native Windows or Linux is preferred | if used, keep the repository inside the Linux file system (not under `/mnt/c`) |
| laptop GPU or power-saving mode | lower power limits distort results | plug in, select maximum performance; mention it in the report to the human |
| interrupted run | sleep, Ctrl+C, reboot, killed command | run the same command again; it resumes |

Setting an environment variable: bash `NAME=value command` (or `export NAME=value; command`);
PowerShell `$env:NAME = "value"; command`.

## Expected timeline

| Step | Time |
|---|---|
| uv + `uv sync` | 2-15 min (download of about 4-5 GB on Linux, about 3 GB on Windows) |
| `doctor` | under 1 min |
| `fetch` | 2-15 min (3.3 GB) |
| `quick` | about 10-15 min |
| `full` (right after quick) | about 20-35 min more (quick + full: about 35-50 min) |
