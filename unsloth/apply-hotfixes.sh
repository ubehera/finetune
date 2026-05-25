#!/usr/bin/env bash
# Re-applies in-venv hotfixes after `uv sync` (which reinstalls unsloth and
# clobbers them). Idempotent — running twice is fine.
#
# Why these hotfixes exist (the saga):
#
# 1. patch_trl_tokenizer_processing_class — Unsloth's monkey-patcher for the
#    backward-compat tokenizer= kwarg generates `model = <class 'inspect._empty'>`
#    when trl's SFTTrainer has required (no-default) args. Skip the shim
#    entirely; modern code uses processing_class= directly.
#
# 2. patch_sft_trainer_tokenizer — Unsloth tries to patch
#    `SFTTrainer._prepare_non_packed_dataloader`, which was removed in modern
#    trl (>=0.16). Skip when the attribute is missing.
#
# 3. Accelerator.prepare exec scope — Unsloth re-execs accelerate's prepare()
#    in unsloth's globals, which lack FP8BackendType / AcceleratorState / etc.
#    Use accelerate.accelerator.__dict__ as the exec globals.
#
# 4. _unsloth_get_batch_samples signature — Modern transformers Trainer passes
#    `device` as a 4th positional. Original sig only takes 3. Accept *args/**kw.
#
# 5. xformers disable on Blackwell — xformers attention kernels require
#    capability <= 9 (Ampere/Ada/Hopper). On sm_100/sm_120 they raise
#    NotImplementedError. Force PyTorch SDPA fallback.
#
# Plus a torchvision stub at ../stubs/torchvision/ — there's no aarch64+cu130
# wheel and stable PyPI torchvision has an ABI mismatch with torch 2.13 nightly.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOK="$DIR/.venv/lib/python3.12/site-packages/unsloth/tokenizer_utils.py"
UTILS="$DIR/.venv/lib/python3.12/site-packages/unsloth/models/_utils.py"
LLAMA="$DIR/.venv/lib/python3.12/site-packages/unsloth/models/llama.py"

for f in "$TOK" "$UTILS" "$LLAMA"; do
  if [ ! -f "$f" ]; then
    echo "[apply-hotfixes] missing $f — run 'uv sync' first" >&2
    exit 1
  fi
done

apply_python_patch() {
  local label="$1" file="$2" sentinel="$3"
  shift 3
  if grep -qF "$sentinel" "$file"; then
    echo "[apply-hotfixes] $label: already applied"
    return 0
  fi
  # Run the patch; if it exits non-zero due to "anchor missing" that means the
  # upstream unsloth version fixed the bug natively — treat as a no-op, not a
  # failure. True errors (syntax errors, wrong file path) still bubble up because
  # the python3 subprocess exits 1 and we log it, but we no longer abort.
  local out rc
  out=$("$@" 2>&1) && rc=0 || rc=$?
  if [ $rc -eq 0 ]; then
    echo "[apply-hotfixes] $label: applied"
  elif echo "$out" | grep -q "anchor missing"; then
    echo "[apply-hotfixes] $label: anchor not found in this unsloth version — likely already fixed upstream, skipping"
  else
    echo "[apply-hotfixes] $label: FAILED (rc=$rc): $out" >&2
    return 1
  fi
}

apply_python_patch "1/5 trl signature codegen" "$TOK" "edgexpert hotfix: in newer trl, some args" \
  python3 -c "
import sys
path = '$TOK'
src = open(path).read()
needle = '    parameters = eval(f\"inspect.signature({trainer_name}).parameters\")\n    if \"tokenizer\" in parameters: return None\n\n    args = {'
if needle not in src:
    sys.exit('anchor missing')
fix = '''    parameters = eval(f\"inspect.signature({trainer_name}).parameters\")
    if \"tokenizer\" in parameters: return None

    # edgexpert hotfix: in newer trl, some args (e.g. \`model\`) have no default.
    # Unsloth\\'s repr of inspect.Parameter.empty produces invalid Python, so skip
    # the backward-compat shim entirely - modern user code uses processing_class=
    # directly anyway.
    import inspect as _ins
    if any(p.default is _ins.Parameter.empty for p in parameters.values()
           if p.kind in (_ins.Parameter.POSITIONAL_OR_KEYWORD, _ins.Parameter.KEYWORD_ONLY)):
        return None

    args = {'''
open(path, 'w').write(src.replace(needle, fix))
"

apply_python_patch "2/5 missing _prepare_non_packed_dataloader" "$TOK" "edgexpert hotfix: in newer trl (>=0.16), this private method was removed" \
  python3 -c "
import sys
path = '$TOK'
src = open(path).read()
needle = '    for function_name, replacer in (\n        (\"_prepare_non_packed_dataloader\", \"def tokenize(element):\",),\n        # (\"_prepare_packed_dataloader\", \"if dataset_text_field is not None\",),\n    ):\n        function = getsource(eval(f\"trl.trainer.sft_trainer.SFTTrainer.{function_name}\"))'
if needle not in src:
    sys.exit('anchor missing')
fix = '''    for function_name, replacer in (
        (\"_prepare_non_packed_dataloader\", \"def tokenize(element):\",),
        # (\"_prepare_packed_dataloader\", \"if dataset_text_field is not None\",),
    ):
        # edgexpert hotfix: in newer trl (>=0.16), this private method was removed.
        # Skip the patch if the attribute is gone - modern TRL handles this internally.
        if not hasattr(eval(\"trl.trainer.sft_trainer.SFTTrainer\"), function_name):
            continue
        function = getsource(eval(f\"trl.trainer.sft_trainer.SFTTrainer.{function_name}\"))'''
open(path, 'w').write(src.replace(needle, fix))
"

apply_python_patch "3/5 accelerator.prepare globals" "$UTILS" "edgexpert hotfix: exec in accelerate.accelerator" \
  python3 /dev/stdin "$UTILS" << 'PYEOF3'
import sys
path = sys.argv[1]
src = open(path).read()
# The f-string in _utils.py contains the literal two-char sequence \n (not newline)
needle = (
    "prepare = prepare.replace(x, f'self.state.distributed_type = DistributedType.NO\\n{s}{x}', 1)\n"
    "exec(prepare, globals())\n"
    "accelerate.accelerator.Accelerator.prepare = prepare"
)
if needle not in src:
    sys.exit('anchor missing')
fix = (
    "prepare = prepare.replace(x, f'self.state.distributed_type = DistributedType.NO\\n{s}{x}', 1)\n"
    "# edgexpert hotfix: exec in accelerate.accelerator's own namespace so symbols\n"
    "# the original prepare() references (FP8BackendType, AcceleratorState, etc.)\n"
    "# are in scope. Exec'ing into our globals() drops all those imports.\n"
    "_prepare_globals = dict(accelerate.accelerator.__dict__)\n"
    'exec(prepare, _prepare_globals)\n'
    'accelerate.accelerator.Accelerator.prepare = _prepare_globals["prepare"]'
)
open(path, 'w').write(src.replace(needle, fix))
PYEOF3
apply_python_patch "4/5 batch_samples signature" "$UTILS" "edgexpert hotfix: modern transformers Trainer.get_batch_samples" \
  python3 -c "
import sys
path = '$UTILS'
src = open(path).read()
needle = 'def _unsloth_get_batch_samples(self, epoch_iterator, num_batches):\n    batch_samples = []\n    num_items_in_batch = None'
if needle not in src:
    sys.exit('anchor missing')
fix = '''def _unsloth_get_batch_samples(self, epoch_iterator, num_batches, device=None, *args, **kwargs):
    # edgexpert hotfix: modern transformers Trainer.get_batch_samples passes
    # \`device\` (positional). Original signature only took (epoch_iterator,
    # num_batches), causing TypeError. Accept and ignore the extras.
    batch_samples = []
    num_items_in_batch = None'''
open(path, 'w').write(src.replace(needle, fix))
"

apply_python_patch "5/5 xformers disable on Blackwell" "$LLAMA" "edgexpert hotfix: xformers attention kernels" \
  python3 -c "
import sys
path = '$LLAMA'
src = open(path).read()
needle = 'from triton import __version__ as triton_version\nHAS_XFORMERS = xformers is not None\nBlockDiagonalCausalMask'
if needle not in src:
    sys.exit('anchor missing')
fix = '''from triton import __version__ as triton_version
HAS_XFORMERS = xformers is not None
# edgexpert hotfix: xformers attention kernels require compute capability <= 9
# (Ampere/Ada/Hopper). On Blackwell (sm_100, sm_120) they raise NotImplementedError
# at runtime. Force PyTorch SDPA fallback by disabling xformers on these GPUs.
import torch as _torch_for_cap_check
if HAS_XFORMERS and _torch_for_cap_check.cuda.is_available() and \\\\\\\\
   _torch_for_cap_check.cuda.get_device_capability(0)[0] >= 10:
    HAS_XFORMERS = False
BlockDiagonalCausalMask'''
open(path, 'w').write(src.replace(needle, fix))
"

echo "[apply-hotfixes] all 5 hotfixes processed"
