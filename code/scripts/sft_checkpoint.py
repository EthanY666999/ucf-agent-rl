"""Atomic SFT recovery state; saving does not consume training RNG."""
import random
from pathlib import Path


def save(path,model,optimizer,step,config,elapsed):
    import torch
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
               config=config,elapsed_seconds=elapsed,torch_rng=torch.get_rng_state(),
               python_rng=random.getstate(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);tmp.replace(path)


def restore(path,model,optimizer,config):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['config']==config,'Cannot resume SFT after a protocol/data/model change'
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step'],state['elapsed_seconds']
