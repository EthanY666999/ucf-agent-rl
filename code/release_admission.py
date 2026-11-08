"""Release gates only; this module does not implement or modify UCF."""


def ucf_release_gate(cfg, method):
    if method != 'ucf':
        return None
    version = cfg.get('ucf_version', 'v2')
    if version not in {'v2.2', 'v2.2-ff'}:
        return 'Legacy UCF v2/v2.1 is historical source only, not a runnable release configuration'
    return f'UCF {version} awaits verified 2B acceptance and matching source import; no legacy fallback'


def require_ucf_release(cfg, method):
    reason = ucf_release_gate(cfg, method)
    if reason:
        raise RuntimeError(reason)
