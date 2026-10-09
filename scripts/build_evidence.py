"""Render Figure 4, Table 12, and Supplementary Tables S1--S5 from saved results.
No simulation, new sampling, or exact parameter sweep is performed.

Input tables are preserved. All displays record explicit estimands and precision.
"""
from pathlib import Path
import csv, html, json, sys, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch

ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / ('study2' if (ROOT/'study2').exists() else 'reproducibility/actor_review')
CORE = ROOT / ('results' if (ROOT/'results/learning_summary.csv').exists() else 'reproducibility/core_results')

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, text):
    path.write_text(text, encoding='utf-8', newline='\n')

def esc(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')

def pm(r, key, ci=None, factor=1, digits=5):
    ci = ci or key + '_ci95'
    return f"${float(r[key])*factor:.{digits}f} \\pm {float(r[ci])*factor:.{digits}f}$"

WORLD = dict(no_echo='No echo', high_echo='High echo', both_dependences='Both dependences', echo_shift='Echo shift', source_asymmetry='Source asymmetry', strong_humans='Stronger humans')
POLICY = dict(Full_trace='Paired observation', Randomized_exposure='Randomized exposure', Ignore_echo='Omit echo ablation', Final_uniform='Final-outcome sampling', **{'exposed_vote:A:k3':'Fixed exposed A+3A','blind_vote:A:k3':'Fixed blind A+3A'})
ABBR = dict(Full_trace='Paired', Randomized_exposure='Randomized', Ignore_echo='Omit echo', Final_uniform='Final only', **{'exposed_vote:A:k3':'Fixed exposed','blind_vote:A:k3':'Fixed blind'})
MEM = dict(reset='Reset', cumulative='Cumulative', window8='Window 8', none='Fixed')

# Reuse the 21-point exact illustrative grid already saved in the experiment.
saved_curves = rows(ACTOR / 'results/illustrative_curve.csv')
eta = np.array(sorted({float(r['eta']) for r in saved_curves}))
assert len(eta) == 21
curves = {}
for family in ('exposed_vote', 'blind_vote'):
    records = sorted((r for r in saved_curves if r['topology']==family+':A:k3'), key=lambda r:float(r['eta']))
    assert len(records)==len(eta)
    curves[family]={metric:np.array([float(r[metric]) for r in records]) for metric in ('error','agreement')}
phase = rows(ACTOR / 'results/phase.csv')
grid = [r for r in phase if float(r['p_h']) == .2 and float(r['p_a']) == .1 and float(r['human_agent_price_ratio']) == 4 and float(r['blind_overhead']) == .05 and float(r['budget']) == 3 and r['regime'] == 'hard']
assert len(grid) == 25
names = ['exposed_vote:A:k6','blind_vote:A:k6','echo_gate:H:k1','blind_gate:H:k1']
codes = ['Exposed vote: A + 6A','Blind vote: A + 6A','Exposed gate: H + 1A','Blind gate: H + 1A']
values = np.empty((5,5), dtype=int)
for r in grid:
    values[round(float(r['rho'])*4),round(float(r['eta'])*4)] = names.index(r['winner'])
with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':10, 'axes.labelsize':9, 'legend.fontsize':8, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'}):
    fig = plt.figure(figsize=(7.15,4.35), layout='constrained')
    gs = fig.add_gridspec(2,2,height_ratios=[1,1.15])
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1]); ax4.axis('off')
    for ax,metric,ylabel in [(ax1,'error','Final decision error (%)'),(ax2,'agreement','Unanimous agreement (%)')]:
        for f,color,marker,style,label in [('exposed_vote','#0067A5','o','-','Exposed vote'),('blind_vote','#B35C00','s','--','Blind vote')]:
            ax.plot(eta,100*curves[f][metric],color=color,marker=marker,markevery=4,markersize=4,linestyle=style,label=label)
        ax.set(xlabel=r'Copying probability $\eta_A$',ylabel=ylabel,xlim=(0,1))
        ax.set_xticks([0,.25,.5,.75,1]); ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    ax1.set_ylim(0,11); ax2.set_ylim(0,104)
    ax1.set_title('(a) More agreement, greater error',loc='left'); ax2.set_title('(b) Public agreement',loc='left'); ax1.legend(loc='upper left')
    palette=['#E0EBF4','#A4C5DF','#FAE9D4','#EAC196']
    ax3.imshow(values,origin='lower',cmap=ListedColormap(palette),vmin=-.5,vmax=3.5,aspect='auto',interpolation='nearest')
    hatches = ['', '///', 'xx', '..']
    for j in range(5):
        for i in range(5):
            ax3.add_patch(Rectangle((i-.5,j-.5),1,1,facecolor='none',edgecolor='#404040',linewidth=.4,hatch=hatches[values[j,i]]))
    ax3.set_xticks(range(5),['0','.25','.5','.75','1']); ax3.set_yticks(range(5),['0','.25','.5','.75','1'])
    ax3.set(xlabel=r'Common copying probability $\eta$',ylabel=r'Private dependence $\rho_A$ $\uparrow$')
    ax3.set_title('(c) Best arrangement in the catalog',loc='left')
    ax4.text(0,.98,'Selected review arrangement',va='top',weight='bold',fontsize=10)
    handles=[Patch(facecolor=palette[i],edgecolor='#404040',hatch=hatches[i],label=codes[i]) for i in range(4)]
    ax4.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.025,.85),frameon=False,fontsize=8.6,handleheight=1.4,labelspacing=.7)
    ax4.text(0,.16,'A: agent; H: human. Proposer + reviewers.\nGates reserve a separate human arbiter.\nMinimum error; cost breaks error ties.',va='top',linespacing=1.35,fontsize=8.2)
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/f'figures/fig4_actor_review.{ext}',dpi=320,facecolor='white')
    plt.close(fig)
svg_path=ROOT/'figures/fig4_actor_review.svg'
write(svg_path, '\n'.join(line.rstrip() for line in svg_path.read_text(encoding='utf-8').splitlines()).rstrip()+'\n')
with (ROOT/'figures/fig4_curves.csv').open('w',newline='',encoding='utf-8') as f:
    out=csv.writer(f,lineterminator="\n");out.writerow(['eta','exposed_error','blind_error','exposed_agreement','blind_agreement'])
    for i,x in enumerate(eta):out.writerow([x,*[float(curves[p][m][i]) for m in ('error','agreement') for p in ('exposed_vote','blind_vote')]])
with (ROOT/'figures/fig4_phase.csv').open('w',newline='',encoding='utf-8') as f:
    out=csv.DictWriter(f,fieldnames=list(grid[0]),lineterminator="\n");out.writeheader();out.writerows(grid)
write(ROOT/'figures/fig4_provenance.json',json.dumps({'sources':['study2/results/illustrative_curve.csv','study2/results/phase.csv'],'display_grid':'21 exact equally spaced eta values selected after the core experiment and read from its saved illustrative_curve.csv; no new evaluations or sampling','phase_filter':{'q_H':.2,'q_A':.1,'human_agent_ratio':4,'blind_overhead':.05,'budget':3,'regime':'hard'},'dimensions_inches':[7.15,4.35],'uncertainty':'Exact expectations under the specified model; no sampling intervals','matplotlib':matplotlib.__version__,'accessibility':'Color plus line style/markers; categorical heatmap uses redundant hatching and legend, dependence increases upward; source values included'},indent=2))

summary=rows(ACTOR/'results/learning_summary.csv')
primary=[]
for w in WORLD:
    for p in POLICY:
        r=next(r for r in summary if r['world']==w and r['policy']==p and r['memory']==('none' if p.startswith(('exposed_','blind_')) else 'cumulative'))
        primary.append(r)
caption=r'''Study 2: selection error and net value under paid observation. All six environments are shown; learning policies use cumulative memory. Entries are means $\pm$ 95\% Monte Carlo half-widths over 128 trajectories of 48 rounds. Fixed baselines have exact expected-outcome scores under the model. Error is a percentage; hit is the percentage of rounds attaining a deployment-loss minimum within $10^{-10}$.'''
table=['\\begingroup\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\setlength{\\LTcapwidth}{\\linewidth}',r'\begin{longtable}{@{}llrrr@{}}',r'\caption{'+caption+r'}\label{tab:actor-results}\\',r'\toprule Environment & Observation / fixed rule & Error (\%) & Net value & Hit (\%) \\\midrule',r'\endfirsthead',r'\multicolumn{5}{l}{\tablename\ \thetable\ (continued)}\\',r'\toprule Environment & Observation / fixed rule & Error (\%) & Net value & Hit (\%) \\\midrule',r'\endhead',r'\midrule\multicolumn{5}{r}{Continued on next page}\\\endfoot',r'\bottomrule\endlastfoot']
for i,r in enumerate(primary):
    fixed=r['memory']=='none'
    er=f"${100*float(r['error']):.3f}$" if fixed else pm(r,'error',factor=100,digits=3)
    nv=f"${float(r['net']):.5f}$" if fixed else pm(r,'net')
    world=WORLD[r['world']] if i%6==0 else ''
    table.append(f"{world} & {ABBR[r['policy']]} & {er} & {nv} & {100*float(r['hit']):.2f} \\")
    table[-1]+='\\' + ('*' if i%6!=5 else '') # Keep each environment's six rows together.
    if i%6==5 and i<len(primary)-1:table.append(r'\addlinespace[3pt]')
table.extend([r'\end{longtable}',r'\endgroup'])
write(ROOT/'sections/actor_results.tex','\n'.join(table)+'\n')

# Tables are included in the main PDF after the appendices.
tex=[r'''% Included by main.tex after the appendices.
\begingroup
\renewcommand{\arraystretch}{1.08}
\renewcommand{\footnotesize}{\fontsize{8}{10}\selectfont}
\section*{Supplementary Tables}
\label{sec:supplementary-tables}
\addcontentsline{toc}{section}{Supplementary Tables}
Tables S1--S5 report Study 1 (evidence acquisition and retention), Study 2 (review arrangements and evidence generation), and the supplementary audit controls. Study 2 and the acquisition-matched controls are exploratory. Source paths below are relative to the repository root. Every uncertainty entry is a 95\% Monte Carlo half-width conditional on the supplied simulation parameters. Rounding to zero does not imply a general absence of uncertainty. Protocol definitions and cost units differ between the studies and are specified in the manuscript. Supplementary table panels retain all rows from the named analyses; no outcome-based row filtering is used.
''']
html_parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Supplementary Tables S1–S5</title><style>body{font:15px/1.5 system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 22px;color:#202530}h1,h2,h3{color:#162f45}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{border-bottom:1px solid #ccc;padding:6px 9px;text-align:left;white-space:nowrap}th{background:#e8eef3;position:sticky;top:0}tr:nth-child(even){background:#f6f8fa}code{font-size:12px}</style><h1>Supplementary Tables S1–S5</h1><p>All uncertainty entries are 95% Monte Carlo half-widths conditional on the model. No outcome-based row exclusion. The PDF defines symbols and estimands; full-precision input tables accompany the reproducibility package.</p>']
index=[]

SOURCES={'S1':['results/learning_summary.csv','results/learning_contrasts.csv'],'S2':['results/learning_conditional.csv'],'S3':['results/robustness_summary.csv'],'S4':['results/audit_extensions_summary.csv'],'S5':['study2/results/learning_summary.csv','study2/results/paired_contrasts.csv']}

def start(number,title,description):
    description = r'{\footnotesize Source: '+ '; '.join(r'\texttt{'+esc(x)+'}' for x in SOURCES[number]) + r'.}\par ' + description
    page_break = '' if number == 'S1' else '\\clearpage\n'
    tex.append(page_break+'\\subsection*{Table '+number+': '+title+'}\n\\label{supp:'+number+'}\n\\addcontentsline{toc}{subsection}{Table '+number+': '+title+'}\n'+description+'\n')
    html_parts.append('<h2 id="'+number+'">Table '+number+': '+html.escape(title)+'</h2><p>'+html.escape(description.replace('\\%','%'))+'</p>')

def panel(title,headers,data,widths=None):
    # Keep the final contrast heading with its table and avoid short continuation pages.
    if title.startswith('E.'): tex.append(r'\clearpage')
    # Tables are split by metric rather than scaled below readable font sizes.
    tex.append('\\subsubsection*{'+title+'}\n\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}\n')
    if title=='Complete audit grid': tex.append(r'\renewcommand{\arraystretch}{.93}')
    if title=='C. Paired net-value contrasts': tex.append(r'\renewcommand{\arraystretch}{1.0}')
    spec=widths or ('l'*len(headers))
    h=' & '.join(headers)+r' \\'
    tex.extend([r'\begin{longtable}{@{}'+spec+r'@{}}',r'\toprule '+h+r'\midrule\endfirsthead',r'\toprule '+h+r'\midrule\endhead',r'\midrule\multicolumn{'+str(len(headers))+r'}{r}{Continued on next page}\\\endfoot',r'\bottomrule\endlastfoot'])
    tex.extend(' & '.join(map(str,row))+r' \\' for row in data)
    tex.extend([r'\end{longtable}\endgroup'])
    html_parts.append('<h3>'+html.escape(title)+'</h3><div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>')
    for row in data:
        vals=[str(v).replace('$','').replace(r'\pm','±').replace(r'\%','%').replace(r'\_','_') for v in row]
        html_parts.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    html_parts.append('</tbody></table></div>')
    index.append({'panel':title,'rows':len(data),'columns':headers})

env_short={'Stationary harm':'Stationary','Workflow reversal':'Reversal','Uniform gain':'Uniform gain'}
arm_short={'Successive rejection':'Rejection','Successive rejects':'Rejection','Discover once':'Once','Repeated discovery':'Repeated'}
core=rows(CORE/'learning_summary.csv')
def core_id(r):return [env_short.get(r['environment'],r['environment']),r['memory'],arm_short.get(r['arm'],r['arm'])]
start('S1','Study 1: evidence acquisition and retention',r'Core design: 54 conditions, each with 128 independent replicate trajectories of 48 rounds. Window 8 includes eight previous rounds plus current observations. Net value, regret and expenses are per production task. Harm is the percentage of rounds selecting a template worse than standard. Late net value covers the final eight rounds. Panel C separates core and exploratory control contrasts; intervals use paired replicate differences.')
panel('A. Net value, harmful adoption, and selection regret',['Environment','Memory','Arm','Net value',r'Harm (\%)','Regret'],[core_id(r)+[pm(r,'net'),pm(r,'harm',factor=100,digits=3),pm(r,'regret')] for r in core])
panel('B. Evaluation expense, program-trial expense, and late outcome',['Environment','Memory','Arm','Expense','Trial expense','Late net'],[core_id(r)+[pm(r,'expense'),pm(r,'meta_expense'),pm(r,'late_net')] for r in core])
contrasts=rows(CORE/'learning_contrasts.csv')
contrast_env={'Workflow reversal':'Reversal at 25','Workflow reversal 20':'Reversal at 21','Workflow reversal 28':'Reversal at 29','Stationary harm':'Stationary','Uniform gain':'Uniform gain'}
tex.append('Reversal labels in Panel C give the one-based round before which the change occurs.\n')
panel('C. Paired net-value contrasts',['Study','Environment','Memory','Comparison','$N$','Difference'],[[esc(r['study']),contrast_env[r['environment']],r['memory'],esc(r['comparison']),r['n'],pm(r,'delta','ci95')] for r in contrasts],r'lp{.19\linewidth}lp{.29\linewidth}rr')

condition=rows(CORE/'learning_conditional.csv')
start('S2','Retained programs and conditional block outcomes',r'All 189 observed program groups are retained: three environments and three memories, both discover-once and repeated-discovery rules, and all three retained programs at their scheduled reviews. Shares use 128 runs per review condition; $n$ is the number retaining the named program. Net and harm cover the subsequent eight-round block. Harm is expressed as a percentage. These are descriptive, endogenously selected groups; no causal program effect or unavailable conditional interval is implied. Every reported group has at least one retained run.')
panel('Complete conditional summary',['Environment','Memory','Arm','Round','Program','$n$',r'Share (\%)','Net',r'Harm (\%)'],[core_id(r)+[r['review_round'],r['program'],r['n'],f"{100*float(r['share']):.2f}",f"{float(r['net']):.5f}",f"{100*float(r['harm']):.3f}"] for r in condition])

audit=rows(CORE/'robustness_summary.csv')
start('S3','Audit policy and memory-window grid',r'All 54 combinations of three environments, three memory windows, and six audit policies. Each cell uses 200 independent seeds. Paired differences compare with fixed audit rate .10 within the same environment and window. The outcome is net value per task under the supplementary audit model.')
panel('Complete audit grid',['Environment','Window','Policy','Net value','Difference vs. fixed .10'],[[esc(r['scenario']),r['window'],esc(r['policy']),pm(r,'mean','ci95_halfwidth'),pm(r,'difference_vs_fixed010','paired_ci95_halfwidth')] for r in audit])

evsi=rows(CORE/'audit_extensions_summary.csv')
start('S4','Expected-value-of-information sensitivity',r'All 72 combinations of three priors, two routing thresholds, two candidate-generation costs, two environments, and three policies. Each cell uses 96 independent test seeds; development seeds selected the horizon separately. Nonzero generation cost charges counterfactual A outputs when route B is executed. Paired differences use fixed audit rate .10 within the same condition.')
panel('Complete prior, threshold, and generation-cost grid',['Prior','Threshold','$g_c$','Environment','Policy','Net value','Difference vs. fixed .10'],[[r['prior'],r['threshold'],r['candidate_generation_cost'],esc(r['scenario']),esc(r['policy']),pm(r,'mean','ci95'),pm(r,'difference_vs_fixed010','paired_ci95')] for r in evsi])

start('S5','Study 2: review arrangements and evidence generation',r'All 84 environment--policy--memory conditions: six environments, four paid policies crossed with three memories, and two fixed baselines per environment. Each learning condition uses 128 trajectories of 48 rounds; late outcomes cover the final 16 rounds. Error and optimal-choice hit are percentages. Regret is deployment loss above the finite-catalog minimum, excluding acquisition and switching. Acquisition cost is per round; production cost is per task in human-call units. Fixed baselines have deterministic expected-outcome scores; their displayed zero half-widths describe the simulation, not real-world certainty. Paired, Randomized, Omit echo and Final only denote the four observation policies defined in Appendix E.')
def actor_id(r):return [WORLD[r['world']],ABBR[r['policy']],MEM[r['memory']]]
panel('A. Full-horizon selection and net value',['Environment','Policy','Memory',r'Error (\%)','Net value',r'Hit (\%)'],[actor_id(r)+[pm(r,'error',factor=100,digits=3),pm(r,'net'),pm(r,'hit',factor=100,digits=2)] for r in summary])
panel('B. Full-horizon costs and regret',['Environment','Policy','Memory','Acquisition cost','Production cost','Regret'],[actor_id(r)+[pm(r,'acquisition_cost',digits=3),pm(r,'production_cost',digits=4),pm(r,'regret')] for r in summary])
panel('C. Final-16-round outcomes',['Environment','Policy','Memory',r'Late error (\%)','Late net',r'Late hit (\%)'],[actor_id(r)+[pm(r,'error_late',factor=100,digits=3),pm(r,'net_late'),pm(r,'hit_late',factor=100,digits=2)] for r in summary])
panel('D. Final-16-round costs and regret',['Environment','Policy','Memory','Late acquisition','Late production','Late regret'],[actor_id(r)+[pm(r,'acquisition_cost_late',digits=3),pm(r,'production_cost_late',digits=4),pm(r,'regret_late')] for r in summary])
pairs=rows(ACTOR/'results/paired_contrasts.csv')
panel('E. Paired policy and memory contrasts',['Environment','Policy A / memory','Policy B / memory','Error diff. (pp)','Net difference','Regret diff.'],[[WORLD[r['world']],ABBR[r['policy_a']]+' / '+MEM[r['memory_a']],ABBR[r['policy_b']]+' / '+MEM[r['memory_b']],pm(r,'error_difference','error_ci95',100,3),pm(r,'net_difference','net_ci95'),pm(r,'regret_difference','regret_ci95')] for r in pairs],r'l>{\raggedright\arraybackslash}p{.175\linewidth}>{\raggedright\arraybackslash}p{.175\linewidth}rrr')
tex.append(r'\endgroup')
html_parts.append('</html>')
write(ROOT/'supplement/supplementary_tables.tex','\n'.join(tex)+'\n')
write(ROOT/'supplement/supplementary_tables.html','\n'.join(html_parts))
write(ROOT/'supplement/table_inventory.json',json.dumps({'panels':index,'sources':SOURCES,'tag':'v2'},indent=2))
print('Built Figure 4, all 36 cumulative/fixed rows for Table 12, and',len(index),'supplementary table panels.')
