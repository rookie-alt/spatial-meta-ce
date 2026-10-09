import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.meta import adapt, forward, meta_loss
from spatial_meta_ce.model import SetEstimator

assert os.environ['CUDA_VISIBLE_DEVICES'] == '6'
assert torch.cuda.device_count() == 1
info = subprocess.check_output(['nvidia-smi', '--id=6', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
uuid, memory, utilization = [part.strip() for part in info.split(',')]
if int(memory) > 64 or int(utilization) > 0:
    raise RuntimeError('GPU6 is no longer idle; refusing smoke')
torch.set_num_threads(1)
torch.manual_seed(2718)
def sample(i, truth):
    return {'groups': [torch.rand(2, 40), torch.rand(2, 40), torch.rand(2, 4), torch.rand(2, 43)], 'true': truth, 'pg': 2.0, 'id': str(i)}
support = batch_samples([sample(i, 200.0) for i in range(5)], 'cuda')
query = batch_samples([sample(i + 100, 300.0) for i in range(64)], 'cuda')
records = []
for name, scope, steps, eval_scope in [('A_all5_head3', 'all', 5, 'head'), ('B_head3_head3', 'head', 3, 'head'), ('C_last3_last3', 'last', 3, 'last')]:
    torch.manual_seed(42)
    model = SetEstimator(arch='sum', width=64).to('cuda')
    initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    started = time.monotonic()
    for _ in range(20):
        optimizer.zero_grad()
        loss = meta_loss(model, support, query, 1e-3, steps, scope)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    params = adapt(model, support, 1e-3, 3, eval_scope, training=False)
    with torch.no_grad():
        prediction = forward(model, params, query)
        assert torch.isfinite(prediction).all()
    torch.cuda.synchronize()
    records.append({'arm': name, 'initial_sha256': initial, 'steps': 20, 'seconds': time.monotonic()-started, 'synthetic_loss': float(loss.detach()), 'finite_predictions': True})
assert len({r['initial_sha256'] for r in records}) == 1
result = {'status': 'passed', 'scope': 'synthetic_gpu_smoke_not_formal_experiment', 'gpu_uuid': uuid, 'torch': torch.__version__, 'arms': records}
Path('reports/meta65-source-screen-20261009/gpu-smoke.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
