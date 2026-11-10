"""Render recorded metrics only; no fitting, smoothing or parameter selection."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(data_path,output,title,preliminary=False):
    data=json.loads(Path(data_path).read_text());rows=sorted(data['curve_rows'],key=lambda r:r['step'])
    steps=[r['step'] for r in rows]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    def line(ax,values,label):
        if any(v is not None for v in values):ax.plot(steps,[float('nan') if v is None else v for v in values],label=label,linewidth=1.2)
    specs=[('Success (%)','success_rate',100),('Response length (tokens)','mean_response_tokens',1),('Hole termination (%)',None,1),('Step duration (s)','step_seconds',1),('KL','kl',1),('Gradient norm before clipping','grad_norm',1)]
    for ax,(label,key,scale) in zip(axes.flat,specs):
        if key:values=[r.get(key)*scale if r.get(key) is not None else None for r in rows]
        else:values=[100*r.get('terminal_counts',{}).get('hole',0)/r['episodes'] if r.get('episodes') else None for r in rows]
        line(ax,values,'Training');ax.set(xlabel='Training step',ylabel=label);ax.grid(alpha=.2)
    validation=data.get('validation',{})
    pairs=sorted((int(k.split('_')[-1]),v['success_rate_pct']) for k,v in validation.items())
    if pairs:axes[0,0].plot(*zip(*pairs),marker='o',label='Fixed validation',color='#cb6124');axes[0,0].legend()
    fig.suptitle(title+(' — IN PROGRESS' if preliminary else '')+'\nUnsmoothed recorded metrics; test sets excluded from training curves')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.pdf']:fig.savefig(output.with_suffix(suffix))
    plt.close(fig)
    if any('ucf_preupdate_fullbatch_components' in r for r in rows):
        fig,axes=plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
        groups=[('Initial group fractions',[(g,lambda r,g=g:r.get('initial_group_fractions',{}).get(g)) for g in ['all_failure','all_success','mixed']]),
                ('Rescue rate',[('rescue',lambda r:r.get('rescue_rate'))]),
                ('Flip rate by group',[(g,lambda r,g=g:r.get('flip_rate_by_group',{}).get(g)) for g in ['all_failure','all_success','mixed']]),
                ('Parse mismatch fractions',[('original',lambda r:r.get('original_parse_mismatch_fraction')),('counterfactual',lambda r:r.get('counterfactual_parse_mismatch_fraction'))]),
                ('Fork-count quantiles',[(name,lambda r,i=i:r.get('forks_p10_p50_p90',[None,None,None])[i]) for i,name in enumerate(['p10','p50','p90'])]),
                ('Mean forks per task',[('mean',lambda r:r.get('mean_forks'))]),
                ('Part B gradient norm',[('B',lambda r:r.get('ucf_preupdate_fullbatch_components',{}).get('B',{}).get('grad_norm'))]),
                ('KL',[('KL',lambda r:r.get('kl'))])]
        for ax,(label,series) in zip(axes.flat,groups):
            for name,fn in series:line(ax,[fn(r) for r in rows],name)
            ax.set(xlabel='Training step',ylabel=label);ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=8)
        fig.suptitle(title+' — UCF diagnostics'+(' (IN PROGRESS)' if preliminary else ''))
        for suffix in ['.png','.pdf']:fig.savefig(output.with_name(output.name+'_ucf').with_suffix(suffix))
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--title',required=True);p.add_argument('--preliminary',action='store_true');a=p.parse_args()
    render(a.data,a.output,a.title,a.preliminary)
