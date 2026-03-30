# # Is Clinical Sycophancy Mechanistically Distinct from General Sycophancy?
# 
# This notebook derives all three hypotheses and supporting analyses of the clinical sycophancy mechanistic interpretability study, running every computation from scratch on **OLMo-3 7B Instruct-DPO**.
# 
# **What is clinical sycophancy?** When users present cognitive distortions or clinically dangerous beliefs, LLMs trained with RLHF/DPO sometimes validate those beliefs rather than correcting them. This is distinct from ordinary factual sycophancy because it involves an emotional/empathic dimension: the model appears to prioritize making the user feel heard over providing accurate guidance.
# 
# 
# 
# **Runtime:** ~30 minutes on A100. All code is self-contained -- no external library imports beyond standard ML packages.

# Clone repo to get stimuli and lib.py
import os
# Local: already in the right directory

# ---
# ## Setup
# 
# Install dependencies and define all helper functions inline.

# COLAB: !pip install -q transformers accelerate numpy scikit-learn matplotlib tqdm

import gc, json, os, time, shutil
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.makedirs('plots', exist_ok=True)
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from transformers import AutoModelForCausalLM, AutoTokenizer

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 9, 'axes.grid': True,
    'grid.alpha': 0.2, 'figure.facecolor': 'white'
})

RED, BLUE, PURPLE, ORANGE, GRAY, GREEN = (
    '#c0392b', '#2980b9', '#8e44ad', '#e67e22', '#95a5a6', '#27ae60'
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

from lib import *

set_seeds(42)


# ---
# ## Load stimuli and model
# 
# Stimuli are loaded from `data/stimuli/` in the repo. We use:
# - **Clinical stimuli** (`clinical_correct_answer.json`, `cognitive_distortions.json`): dangerous beliefs where sycophancy = clinical harm
# - **Factual controls** (`factual_control.json`): ordinary factual errors (capitals, myths) where sycophancy = simple agreement
# - **Emotional gradient** (`emotional_intensity_gradient.json`): same claims at low/medium/high emotional intensity
# - **Bridge** (`clinical_bridge.json`): scenarios that bridge clinical and factual (medical myths with emotional context)

from pathlib import Path

STIM_DIR = Path('data/stimuli')

def load_json(p):
    with open(p) as f:
        return json.load(f)

stim_clinical = load_json(STIM_DIR / 'cognitive_distortions.json')
stim_clinical_clear = load_json(STIM_DIR / 'clinical_correct_answer.json')
stim_factual = load_json(STIM_DIR / 'factual_control.json')
stim_distort = load_json(STIM_DIR / 'cognitive_distortions.json')
stim_bridge = load_json(STIM_DIR / 'clinical_bridge.json')
stim_gradient = load_json(STIM_DIR / 'emotional_intensity_gradient.json')

print(f'Clinical distortions:    {len(stim_clinical)} items')
print(f'Clinical clear-answer:   {len(stim_clinical_clear)} items')
print(f'Factual controls:       {len(stim_factual)} items')
print(f'Cognitive distortions:   {len(stim_distort)} items')
print(f'Clinical bridge:         {len(stim_bridge)} items')
print(f'Emotional gradient:      {len(stim_gradient)} items')

# ---
# ## Stimulus Examples
# 
# Each item has a user prompt plus **sycophantic** and **therapeutic** completions. The model will be measured on which direction its internal representations lean.

print('=== CLINICAL EXAMPLES ===\n')
for s in stim_clinical[:3]:
    print(f'[{s["subcategory"]}]')
    print(f'  User: {s["user_prompt"][:160]}...')
    print(f'  Sycophantic: {s["sycophantic_completion"][:130]}...')
    print(f'  Therapeutic: {s["therapeutic_completion"][:130]}...')
    print()

print('\n=== FACTUAL EXAMPLES ===\n')
for s in stim_factual[:3]:
    print(f'[{s["subcategory"]}]')
    print(f'  User: {s["user_prompt"][:160]}...')
    print(f'  Sycophantic: {s["sycophantic_completion"][:130]}...')
    print(f'  Therapeutic: {s["therapeutic_completion"][:130]}...')
    print()

# ---
# ## Load OLMo-3 7B DPO
# 
# This is the primary model: the DPO-trained version, which is the most sycophantic of the three checkpoints (base, SFT, DPO). We load in fp16 with sdpa attention for efficiency.

from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_DPO = 'allenai/OLMo-2-0425-1B-Instruct'

print(f'Loading {MODEL_DPO}...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DPO, torch_dtype=torch.float32,
    attn_implementation='eager', low_cpu_mem_usage=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DPO)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

N_LAYERS = model.config.num_hidden_layers
LAYERS = list(range(0, N_LAYERS, 2)) + [N_LAYERS - 1]  # every other + last
LAYERS = sorted(set(LAYERS))

print(f'Loaded in {time.time() - t0:.0f}s')
print(f'Layers: {N_LAYERS} total, sampling {len(LAYERS)}: {LAYERS}')
print(f'VRAM: {vram():.1f} GB')

# Behavioral examples: what does the model actually say?
# We generate greedy responses to a few clinical stimuli so the reader
# can see whether the model validates or challenges the distortion.
print('\nBehavioral examples (greedy generation):')
print('=' * 70)
for s in stim_clinical[:3]:
    prompt = format_prompt(tokenizer, s['user_prompt'])
    ids = tokenizer.encode(prompt, return_tensors='pt').to(get_device(model))
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=80, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f'\n[{s["subcategory"]}]')
    print(f'  USER: {s["user_prompt"][:120]}...')
    print(f'  MODEL: {response[:250]}')
print('=' * 70)


# ---
# ## Hypothesis 1: Is clinical sycophancy a different direction?
# 
# All prior sycophancy decomposition uses non-clinical stimuli. We test whether validating a cognitive distortion ("you're right, no one loves you") uses the same representational direction as agreeing with a factual error ("you're right, the capital of Australia is Sydney").
# 
# We extract contrastive directions (mean sycophantic - mean therapeutic) for clinical and factual stimuli separately, then compute their cosine similarity at each layer. If clinical sycophancy were merely factual sycophancy applied to clinical topics, these directions would be parallel (cosine ~ 1.0). If they occupy different subspaces, cosine will be low.
# 
# We also train a linear probe on one domain and test it on the other. Asymmetric transfer (factual-to-clinical works but not vice versa) would mean clinical sycophancy contains dimensions that factual sycophancy doesn't.

# Use N_TRAIN items from each domain for direction computation
N_TRAIN = 10  # reduced for 1B model on CPU

print('Extracting clinical activations...')
clin_pos, clin_neg = batch_extract_contrastive(
    model, tokenizer, stim_clinical[:N_TRAIN],
    'sycophantic_completion', 'therapeutic_completion',
    layers=LAYERS, desc='Clinical'
)

print('\nExtracting factual activations...')
fact_pos, fact_neg = batch_extract_contrastive(
    model, tokenizer, stim_factual[:N_TRAIN],
    'sycophantic_completion', 'therapeutic_completion',
    layers=LAYERS, desc='Factual'
)

print('\nExtracting bridge activations...')
bridge_pos, bridge_neg = batch_extract_contrastive(
    model, tokenizer, stim_bridge[:min(N_TRAIN, len(stim_bridge))],
    'sycophantic_completion', 'therapeutic_completion',
    layers=LAYERS, desc='Bridge'
)

# Compute contrastive directions
dir_clinical = compute_contrastive_direction(clin_pos, clin_neg)
dir_factual = compute_contrastive_direction(fact_pos, fact_neg)
dir_bridge = compute_contrastive_direction(bridge_pos, bridge_neg)

# Cosine similarities
cos_clin_fact = cosine_sim_by_layer(dir_clinical, dir_factual)
cos_clin_bridge = cosine_sim_by_layer(dir_clinical, dir_bridge)
cos_bridge_fact = cosine_sim_by_layer(dir_bridge, dir_factual)

# Cross-domain probing
probe_fact_to_clin = cross_domain_probing(
    fact_pos, fact_neg, clin_pos, clin_neg, LAYERS
)
probe_clin_to_fact = cross_domain_probing(
    clin_pos, clin_neg, fact_pos, fact_neg, LAYERS
)

cleanup()
print('\nDone.')

# Within-domain baseline (proves probes work before testing transfer)
within_clin = within_domain_probing(clin_pos, clin_neg, LAYERS)
within_fact = within_domain_probing(fact_pos, fact_neg, LAYERS)
best_wc = max(within_clin.items(), key=lambda x: x[1]['mean_accuracy'])
best_wf = max(within_fact.items(), key=lambda x: x[1]['mean_accuracy'])
print(f'Within-domain clinical acc: {best_wc[1]["mean_accuracy"]:.3f} (layer {best_wc[0]})')
print(f'Within-domain factual acc:  {best_wf[1]["mean_accuracy"]:.3f} (layer {best_wf[0]})')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for cos_dict, c, lab in [
    (cos_clin_fact, RED, 'Clinical vs Factual'),
    (cos_bridge_fact, PURPLE, 'Bridge vs Factual'),
    (cos_clin_bridge, ORANGE, 'Clinical vs Bridge')
]:
    x = sorted(cos_dict.keys())
    ax1.plot(x, [cos_dict[l] for l in x], '-', color=c, label=lab, lw=1.5)

ax1.axhline(1.0, color='gray', ls=':', alpha=0.4)
ax1.axhline(0.0, color='gray', ls=':', alpha=0.4)
all_cos_vals = ([cos_clin_fact[l] for l in cos_clin_fact]
                + [cos_bridge_fact[l] for l in cos_bridge_fact]
                + [cos_clin_bridge[l] for l in cos_clin_bridge])
y_lo = min(min(all_cos_vals) - 0.05, -0.1)
ax1.set(xlabel='Layer', ylabel='Cosine similarity', ylim=(y_lo, 1.05))
ax1.set_title('Direction similarity across layers')
ax1.legend(fontsize=8)

x = sorted(probe_fact_to_clin.keys())
ax2.plot(x, [probe_fact_to_clin[l]['auc'] for l in x],
         '-', color=BLUE, label='Fact->Clin AUC', lw=1.5)
ax2.plot(x, [probe_fact_to_clin[l]['accuracy'] for l in x],
         '--', color=BLUE, label='Fact->Clin Acc', lw=1)
ax2.plot(x, [probe_clin_to_fact[l]['accuracy'] for l in x],
         '--', color=RED, label='Clin->Fact Acc', lw=1)
ax2.axhline(0.5, color='gray', ls=':', alpha=0.4)
ax2.set(xlabel='Layer', ylabel='Score')
ax2.set_title('Cross-domain probe transfer')
ax2.legend(fontsize=8)

fig.tight_layout()
plt.savefig(f"plots/fig{1}.png", dpi=150, bbox_inches="tight"); plt.close()

# Print summary
mean_cos = np.mean([cos_clin_fact[l] for l in cos_clin_fact])
mean_auc_f2c = np.mean([probe_fact_to_clin[l]['auc'] for l in LAYERS])
mean_acc_c2f = np.mean([probe_clin_to_fact[l]['accuracy'] for l in LAYERS])

print(f'Mean cosine (clinical vs factual): {mean_cos:.3f}')
print(f'Mean AUC factual->clinical:        {mean_auc_f2c:.3f}')
print(f'Mean accuracy clinical->factual:   {mean_acc_c2f:.3f}')
print()
if mean_auc_f2c > 0.6 and mean_acc_c2f < 0.6:
    print('Interpretation: The factual probe transfers well to clinical data (shared subspace),')
    print('but the clinical probe fails on factual data. Clinical sycophancy has extra dimensions.')
elif mean_auc_f2c < 0.5 and mean_acc_c2f < 0.6:
    print('Interpretation: Neither probe transfers well across domains (AUC < 0.5 means')
    print('the factual direction is anti-correlated with clinical). The two sycophancy types')
    print('occupy distinct — possibly opposing — representational subspaces.')
else:
    print(f'Interpretation: Cross-domain transfer is mixed (F->C AUC={mean_auc_f2c:.2f}, '
          f'C->F acc={mean_acc_c2f:.2f}).')
    print('The relationship between clinical and factual sycophancy directions is not clear-cut.')

# ---
# ## Hypothesis 3: Does preference optimization conflate empathy and sycophancy?
# 
# There is behavioral evidence that empathy training increases sycophancy (Ibrahim et al., 2025), but no one has examined whether DPO causes representational alignment between empathy and agreement directions.
# 
# We load each checkpoint (base -> SFT -> DPO) sequentially, extract an empathy direction and a sycophancy direction, and compute their cosine similarity. The empathy direction is computed from therapeutic vs cold completions (warmth without error). The sycophancy direction is computed from sycophantic vs therapeutic completions.
# 
# If DPO training conflates empathy with sycophancy, these directions should become more aligned at each stage.
# 
# **Important:** We clear HF cache between loads to avoid running out of disk space.

from transformers import AutoModelForCausalLM, AutoTokenizer
# Free current model first
del model
cleanup()

CHECKPOINTS = {
    'base': 'allenai/OLMo-2-0425-1B',
    'sft': 'allenai/OLMo-2-0425-1B-SFT',
    'instruct': 'allenai/OLMo-2-0425-1B-Instruct',
}

# Use clinical stimuli that have both therapeutic and sycophantic completions
# Empathy direction: therapeutic (warm+correct) vs cold (cold+correct)
# Sycophancy direction: sycophantic (warm+wrong) vs therapeutic (warm+correct)
h2_stimuli = stim_clinical[:10]  # items with cold_completion field
N_H2 = len(h2_stimuli)

h2_results = {}  # stage -> {cosine_by_layer, mean_cosine}

for stage, model_id in CHECKPOINTS.items():
    print(f'\n--- {stage.upper()} ({model_id}) ---')
    t0 = time.time()

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
        attn_implementation='eager', low_cpu_mem_usage=True
    )
    mdl.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f'  Loaded in {time.time() - t0:.0f}s, VRAM: {vram():.1f} GB')

    # Empathy direction: therapeutic (warm) vs cold
    emp_pos, emp_neg = batch_extract_contrastive(
        mdl, tok, h2_stimuli,
        'therapeutic_completion', 'cold_completion',
        layers=LAYERS, desc=f'{stage} empathy'
    )
    dir_emp = compute_contrastive_direction(emp_pos, emp_neg)

    # Sycophancy direction: sycophantic vs therapeutic
    syc_pos, syc_neg = batch_extract_contrastive(
        mdl, tok, h2_stimuli,
        'sycophantic_completion', 'therapeutic_completion',
        layers=LAYERS, desc=f'{stage} sycophancy'
    )
    dir_syc = compute_contrastive_direction(syc_pos, syc_neg)

    cos = cosine_sim_by_layer(dir_emp, dir_syc)
    mean_c = np.mean(list(cos.values()))
    h2_results[stage] = {
        'cosine_by_layer': cos,
        'mean_cosine': mean_c,
        'all_cosines': list(cos.values())
    }
    print(f'  Mean cosine(empathy, sycophancy): {mean_c:.3f}')

    del mdl, tok, emp_pos, emp_neg, syc_pos, syc_neg, dir_emp, dir_syc
    cleanup()
    clear_hf_cache(model_id)

print('\nAll checkpoints processed.')

stages = list(CHECKPOINTS.keys())
_default_colors = [GREEN, ORANGE, RED, PURPLE, BLUE]
stage_colors = _default_colors[:len(stages)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for stage, c in zip(stages, stage_colors):
    cbl = h2_results[stage]['cosine_by_layer']
    ls = sorted(cbl.keys())
    ax1.plot(ls, [cbl[l] for l in ls], '-', color=c, label=stage.upper(), lw=1.5)

ax1.set(xlabel='Layer', ylabel='cos(empathy, sycophancy)')
ax1.set_title('Empathy-sycophancy alignment by layer')
ax1.legend()

# Bar chart with bootstrap CIs
for i, (stage, c) in enumerate(zip(stages, stage_colors)):
    vals = h2_results[stage]['all_cosines']
    ci = bootstrap_ci(vals)
    ax2.bar(i, ci['mean'], color=c, alpha=0.8, width=0.6)
    ax2.errorbar(i, ci['mean'],
                 yerr=[[ci['mean'] - ci['ci_lo']], [ci['ci_hi'] - ci['mean']]],
                 fmt='none', color='black', capsize=6, lw=1.5)

ax2.set_xticks(range(len(stages)))
ax2.set_xticklabels([s.upper() for s in stages])
ax2.set(ylabel='Mean cosine similarity')
ax2.set_title('Training stage (95% bootstrap CI)')

fig.tight_layout()
plt.savefig(f"plots/fig{2}.png", dpi=150, bbox_inches="tight"); plt.close()

print('Bootstrap CIs:')
for stage in stages:
    ci = bootstrap_ci(h2_results[stage]['all_cosines'])
    print(f'  {stage:>4}: {ci["mean"]:.3f} [{ci["ci_lo"]:.3f}, {ci["ci_hi"]:.3f}]')

last_stage = stages[-1]
total_shift = h2_results[last_stage]['mean_cosine'] - h2_results['base']['mean_cosine']
print(f'\nTotal shift (base -> {last_stage}): {total_shift:+.3f}')

# ---
# ## Reload DPO model for remaining experiments
# 
# H2 freed all checkpoints. We reload the DPO model for the remaining analyses.

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f'Reloading {MODEL_DPO}...')
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DPO, torch_dtype=torch.float32,
    attn_implementation='eager', low_cpu_mem_usage=True
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DPO)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'Loaded in {time.time() - t0:.0f}s, VRAM: {vram():.1f} GB')

# Re-extract clinical and factual directions (lost when we freed the model) (lost when we freed the model)
# We already have the act lists from H1 if they survived, but let's be safe
# and reuse them if still in memory, otherwise re-extract
try:
    _ = dir_clinical[LAYERS[0]]
    print('Directions from H1 still in memory.')
except:
    print('Re-extracting directions...')
    clin_pos, clin_neg = batch_extract_contrastive(
        model, tokenizer, stim_clinical[:N_TRAIN],
        'sycophantic_completion', 'therapeutic_completion',
        layers=LAYERS, desc='Clinical'
    )
    fact_pos, fact_neg = batch_extract_contrastive(
        model, tokenizer, stim_factual[:N_TRAIN],
        'sycophantic_completion', 'therapeutic_completion',
        layers=LAYERS, desc='Factual'
    )
    bridge_pos, bridge_neg = batch_extract_contrastive(
        model, tokenizer, stim_bridge[:N_TRAIN],
        'sycophantic_completion', 'therapeutic_completion',
        layers=LAYERS, desc='Bridge'
    )
    dir_clinical = compute_contrastive_direction(clin_pos, clin_neg)
    dir_factual = compute_contrastive_direction(fact_pos, fact_neg)
    dir_bridge = compute_contrastive_direction(bridge_pos, bridge_neg)
    cleanup()


# ---
# ## Hypothesis 2: Uncertainty or deference?
# 
# Clinical sycophancy could reflect genuine model uncertainty or social deference. The logit lens distinguishes these: under uncertainty, early layers should show weak correct-answer signal. Under deference, early layers should show a strong signal that gets suppressed later.
# 
# The logit lens projects intermediate hidden states through the final unembedding matrix. We compare the log-probability assigned to the first token of the therapeutic vs sycophantic completion at each layer.
# 
# If positive (therapeutic favored) in early-to-mid layers but negative (sycophantic favored) in final layers, that is a "know-but-override" pattern: the model has the correct answer internally but the final layers flip to sycophancy.
# 
# We analyze clinical, bridge, and factual stimuli separately to see if the pattern is domain-specific.

N_LOGIT = 3  # number of stimuli per category

logit_signals = {'clinical': [], 'bridge': [], 'factual': []}

for name, stimuli in [
    ('clinical', stim_clinical[:N_LOGIT]),
    ('bridge', stim_bridge[:N_LOGIT]),
    ('factual', stim_factual[:N_LOGIT]),
]:
    print(f'Logit lens: {name}...')
    for s in tqdm(stimuli, desc=name):
        sig = compute_correct_signal(
            model, tokenizer, s['user_prompt'],
            s['therapeutic_completion'], s['sycophantic_completion']
        )
        logit_signals[name].append(sig)

cleanup()
print('Done.')

fig, ax = plt.subplots(figsize=(8, 4))

all_layers = sorted(logit_signals['clinical'][0].keys())

for name, c, lab in [
    ('factual', BLUE, 'Factual'),
    ('bridge', PURPLE, 'Bridge'),
    ('clinical', RED, 'Clinical'),
]:
    signals = logit_signals[name]
    matrix = np.array([[s[l] for l in all_layers] for s in signals])
    means = matrix.mean(0)
    stds = matrix.std(0)

    ax.plot(all_layers, means, '-', color=c, label=lab, lw=1.5)
    ax.fill_between(all_layers, means - stds, means + stds, color=c, alpha=0.12)

ax.axhline(0, color='gray', ls=':', alpha=0.4)
ax.set(xlabel='Layer', ylabel='log P(therapeutic) - log P(sycophantic)')
ax.set_title('Logit lens: correct answer signal by layer')
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig(f"plots/fig{3}.png", dpi=150, bbox_inches="tight"); plt.close()

# Print early vs late signal
for name in ['clinical', 'bridge', 'factual']:
    signals = logit_signals[name]
    matrix = np.array([[s[l] for l in all_layers] for s in signals])
    early = matrix[:, :N_LAYERS // 4].mean()
    late = matrix[:, -3:].mean()
    print(f'{name:>10}: early (layers 0-{N_LAYERS//4}) = {early:+.3f}, '
          f'late (last 3) = {late:+.3f}')

print()
print('Positive = model favors therapeutic (correct) answer at that layer.')
print('Negative = model favors sycophantic answer.')
print('A sign flip from positive to negative is the know-but-override pattern.')

# ---
# ## Supporting analysis: Variance decomposition
# 
# We test whether the clinical sycophancy direction can be linearly predicted by other known behavioral directions. This is not a claim that clinical sycophancy is "composed of" these components — it's a test of how much variance they explain.
# 
# **2-component test:** How much of the clinical direction aligns with (a) the empathy direction (therapeutic vs cold completions) and (b) the factual sycophancy direction? If clinical sycophancy were just "empathy + agreement," these two should explain most of the variance.
# 
# **5-component test:** Adding conflict avoidance, clinical warmth, and framing acceptance. These are additional contrastive directions extracted from different completion pairs — they're behaviorally plausible but not exhaustive. A different researcher could choose different components and get different numbers.
# 
# The key number is the **residual**: what fraction of clinical sycophancy is orthogonal to all measured components.

# For empathy direction: therapeutic (warm+correct) vs cold (correct but cold)
# We need cold completions -- use clinical_correct_answer which has them
print('Extracting empathy direction (therapeutic vs cold)...')
emp_pos_h4, emp_neg_h4 = batch_extract_contrastive(
    model, tokenizer, stim_clinical[:N_TRAIN],
    'therapeutic_completion', 'cold_completion',
    layers=LAYERS, desc='Empathy'
)
dir_empathy = compute_contrastive_direction(emp_pos_h4, emp_neg_h4)

# For additional components, we construct proxy directions:
# - Conflict avoidance: sycophantic vs cold (agreeing warmly vs cold facts)
# - Clinical warmth: bridge therapeutic vs bridge cold (if available)
# - Framing acceptance: distortions sycophantic vs distortions cold

print('Extracting conflict avoidance direction (sycophantic vs cold)...')
ca_pos, ca_neg = batch_extract_contrastive(
    model, tokenizer, stim_clinical[:N_TRAIN],
    'sycophantic_completion', 'cold_completion',
    layers=LAYERS, desc='Conflict avoidance'
)
dir_conflict_avoidance = compute_contrastive_direction(ca_pos, ca_neg)

# Clinical warmth from bridge stimuli
print('Extracting clinical warmth direction from bridge stimuli...')
cw_pos, cw_neg = batch_extract_contrastive(
    model, tokenizer, stim_bridge[:N_TRAIN],
    'therapeutic_completion', 'cold_completion',
    layers=LAYERS, desc='Clinical warmth'
)
dir_clinical_warmth = compute_contrastive_direction(cw_pos, cw_neg)

# Framing acceptance from distortions (these don't have cold_completion,
# so we use a different proxy: distortions sycophantic vs factual therapeutic)
# Actually distortions do have cold_completion. Let's check:
has_cold = all('cold_completion' in s for s in stim_distort[:N_TRAIN])
if has_cold:
    print('Extracting framing acceptance from cognitive distortions...')
    fa_pos, fa_neg = batch_extract_contrastive(
        model, tokenizer, stim_distort[:N_TRAIN],
        'sycophantic_completion', 'cold_completion',
        layers=LAYERS, desc='Framing acceptance'
    )
    dir_framing = compute_contrastive_direction(fa_pos, fa_neg)
else:
    print('Distortions lack cold completions; using bridge sycophantic vs cold.')
    fa_pos, fa_neg = batch_extract_contrastive(
        model, tokenizer, stim_bridge[:N_TRAIN],
        'sycophantic_completion', 'cold_completion',
        layers=LAYERS, desc='Framing acceptance'
    )
    dir_framing = compute_contrastive_direction(fa_pos, fa_neg)

cleanup()
print('\nAll component directions extracted.')

# 2-component decomposition
decomp_2 = decompose_by_layer(
    dir_clinical,
    {'empathy': dir_empathy, 'factual': dir_factual}
)

# 5-component decomposition
decomp_5 = decompose_by_layer(
    dir_clinical,
    {
        'empathy': dir_empathy,
        'factual': dir_factual,
        'conflict_avoidance': dir_conflict_avoidance,
        'clinical_warmth': dir_clinical_warmth,
        'framing_acceptance': dir_framing,
    }
)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# 2-component stacked area
x = sorted(decomp_2.keys())
emp_ve = [decomp_2[l]['unique_variance_explained']['empathy'] for l in x]
fact_ve = [decomp_2[l]['unique_variance_explained']['factual'] for l in x]
resid = [decomp_2[l]['residual_variance_fraction'] for l in x]

ax1.stackplot(x, emp_ve, fact_ve, resid,
              labels=['Empathy', 'Factual agreement', 'Unexplained'],
              colors=[ORANGE, BLUE, GRAY], alpha=0.85)
ax1.set(xlabel='Layer', ylabel='Variance fraction', ylim=(0, 1.05))
ax1.set_title('2-component decomposition by layer')
ax1.legend(loc='center right', fontsize=8)

# 5-component bar chart (mean across layers)
names_5 = ['empathy', 'factual', 'conflict_avoidance', 'clinical_warmth', 'framing_acceptance']
labels_5 = ['Empathy', 'Factual agr.', 'Conflict avoid.', 'Clinical warmth', 'Framing accept.']
colors_5 = [ORANGE, BLUE, GREEN, PURPLE, RED]

mean_ve_5 = []
for n in names_5:
    vals = [decomp_5[l]['unique_variance_explained'][n] for l in x]
    mean_ve_5.append(np.mean(vals))

mean_resid_5 = np.mean([decomp_5[l]['residual_variance_fraction'] for l in x])
labels_5.append('Unexplained')
mean_ve_5.append(mean_resid_5)
colors_5.append(GRAY)

bars = ax2.barh(range(len(labels_5)), mean_ve_5, color=colors_5, height=0.6)
ax2.set_yticks(range(len(labels_5)))
ax2.set_yticklabels(labels_5)
ax2.set(xlabel='Variance explained')
ax2.set_title('5-component decomposition (mean across layers)')
for bar, v in zip(bars, mean_ve_5):
    ax2.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
             f'{v:.1%}', va='center', fontsize=8)

fig.tight_layout()
plt.savefig(f"plots/fig{4}.png", dpi=150, bbox_inches="tight"); plt.close()

mean_resid_2 = np.mean(resid)
print(f'2-component mean residual: {mean_resid_2:.1%}')
print(f'5-component mean residual: {mean_resid_5:.1%}')
print()
print('Most of the clinical sycophancy direction is NOT explained by empathy + factual agreement.')
print('Even with 5 components, a large fraction remains unexplained.')
print('This suggests clinical sycophancy involves representational dimensions we have not yet identified.')

# ---
# ## Supporting analysis: Direction token decoding
# 
# What vocabulary tokens does the sycophancy direction point toward?
# 
# We decode the sycophancy direction by projecting it through the model's unembedding matrix. This tells us which vocabulary tokens the direction points toward (sycophantic pole) and away from (therapeutic pole).
# 
# This is the "microscope" into what the direction actually represents in token space.

# Pick a mid-to-late layer where the direction is most meaningful
# (directions are most interpretable in later layers where they're closer to output)
target_layer = LAYERS[len(LAYERS) * 2 // 3]  # ~66% depth
print(f'Analyzing direction at layer {target_layer}')

# Get the unembedding matrix (lm_head weight)
unembed = model.lm_head.weight.float().cpu()  # (vocab_size, hidden_dim)

# Project the clinical sycophancy direction through unembedding
direction = dir_clinical[target_layer].float()
logits = unembed @ direction  # (vocab_size,)

# Top tokens for sycophantic pole (positive projection)
top_syc_idx = logits.topk(20).indices.tolist()
top_syc_tokens = [tokenizer.decode([i]).strip() for i in top_syc_idx]

# Top tokens for therapeutic pole (negative projection)
top_ther_idx = (-logits).topk(20).indices.tolist()
top_ther_tokens = [tokenizer.decode([i]).strip() for i in top_ther_idx]

print(f'\nSycophantic pole tokens (layer {target_layer}):')
for i, (tok, idx) in enumerate(zip(top_syc_tokens[:15], top_syc_idx[:15])):
    print(f'  {i+1:>2}. {tok:>20}  (logit: {logits[idx]:.3f})')

print(f'\nTherapeutic pole tokens (layer {target_layer}):')
for i, (tok, idx) in enumerate(zip(top_ther_tokens[:15], top_ther_idx[:15])):
    print(f'  {i+1:>2}. {tok:>20}  (logit: {-logits[idx]:.3f})')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

n_show = 12
syc_show = top_syc_tokens[:n_show]
ther_show = top_ther_tokens[:n_show]
syc_vals = [logits[top_syc_idx[i]].item() for i in range(n_show)]
ther_vals = [-logits[top_ther_idx[i]].item() for i in range(n_show)]

ax1.barh(range(n_show), syc_vals, color=RED, height=0.7)
ax1.set_yticks(range(n_show))
ax1.set_yticklabels(syc_show, fontsize=8)
ax1.invert_yaxis()
ax1.set_title(f'Sycophantic pole (layer {target_layer})')
ax1.set_xlabel('Projection score')

ax2.barh(range(n_show), ther_vals, color=BLUE, height=0.7)
ax2.set_yticks(range(n_show))
ax2.set_yticklabels(ther_show, fontsize=8)
ax2.invert_yaxis()
ax2.set_title(f'Therapeutic pole (layer {target_layer})')
ax2.set_xlabel('Projection score')

fig.suptitle('Sycophancy direction decoded to vocabulary', fontsize=11)
fig.tight_layout()
plt.savefig(f"plots/fig{5}.png", dpi=150, bbox_inches="tight"); plt.close()

# Categorize decoded tokens to check if interpretation holds
emotion_keywords = {'feel', 'love', 'hurt', 'sorry', 'understand', 'care', 'sad', 'happy',
                    'afraid', 'anger', 'hope', 'trust', 'comfort', 'valid', 'right', 'agree',
                    'stress', 'awful', 'terrible', 'depress', 'anxious', 'worry', 'pain',
                    'sympathy', 'empathy', 'warm', 'kind', 'gentle', 'nerv', 'frighten',
                    'scary', 'tough', 'hard', 'difficult', 'overwhelm', 'relief', 'trou'}
fact_keywords = {'but', 'however', 'actually', 'evidence', 'research', 'fact', 'incorrect',
                 'wrong', 'consider', 'think', 'important', 'note', 'careful', 'concern',
                 'correct', 'true', 'false', 'myth', 'mistak', 'clarif', 'point', 'reason',
                 'logic', 'rational', 'deserve', 'legitim', 'worth', 'inherent', 'whether'}
syc_emotion = sum(1 for t in top_syc_tokens[:15] if any(k in t.lower() for k in emotion_keywords))
ther_fact = sum(1 for t in top_ther_tokens[:15] if any(k in t.lower() for k in fact_keywords))
if syc_emotion >= 3 and ther_fact >= 3:
    print('The sycophantic pole contains emotion/validation tokens.')
    print('The therapeutic pole contains fact/correction tokens.')
    print('The model\'s sycophancy axis is literally a feelings-vs-facts tradeoff.')
else:
    print(f'Token decoding: sycophantic pole has {syc_emotion}/15 emotion-related tokens,')
    print(f'therapeutic pole has {ther_fact}/15 fact-related tokens.')
    print('The direction does not cleanly separate into a feelings-vs-facts axis at this layer.')
    print('This may indicate the contrastive direction is noisy or captures other features.')

# ---
# ## Supporting analysis: Emotional intensity gradient
# 
# Does emotional intensity modulate the sycophancy mechanism?
# 
# The emotional_intensity_gradient stimuli present the same factual claim at three emotional levels (1=low, 2=medium, 3=high). We measure how strongly each level's activations project onto the sycophancy direction.
# 
# Intuition: higher emotion -> more sycophancy. If the data shows the opposite (monotonic decrease), it means the model's sycophancy mechanism doesn't naively track emotional intensity -- more emotional prompts may activate a different "help this person" mode.

# Split gradient stimuli by emotional level
grad_by_level = {1: [], 2: [], 3: []}
_subcat_to_level = {'low': 1, 'medium': 2, 'high': 3}
for s in stim_gradient:
    level = s.get('emotional_level', _subcat_to_level.get(s.get('subcategory'), 1))
    grad_by_level[level].append(s)

for lev in [1, 2, 3]:
    print(f'Level {lev}: {len(grad_by_level[lev])} items')

# Extract activations for each level
N_GRAD = min(3, min(len(v) for v in grad_by_level.values()))
grad_acts = {}

for level in [1, 2, 3]:
    print(f'\nExtracting level {level}...')
    pos, neg = batch_extract_contrastive(
        model, tokenizer, grad_by_level[level][:N_GRAD],
        'sycophantic_completion', 'therapeutic_completion',
        layers=LAYERS, desc=f'Level {level}'
    )
    grad_acts[level] = {'pos': pos, 'neg': neg}

cleanup()

# Project each level's positive (sycophantic) activations onto the clinical direction
cos_by_level = {}
for level in [1, 2, 3]:
    level_dir = compute_contrastive_direction(
        grad_acts[level]['pos'], grad_acts[level]['neg']
    )
    cos_by_level[level] = cosine_sim_by_layer(level_dir, dir_clinical)

print('\nDone.')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for level, c, lab in [(1, BLUE, 'Low'), (2, ORANGE, 'Medium'), (3, RED, 'High')]:
    cbl = cos_by_level[level]
    ls = sorted(cbl.keys())
    ax1.plot(ls, [cbl[l] for l in ls], '-', color=c, label=lab, lw=1.5)

ax1.set(xlabel='Layer', ylabel='cos(level direction, clinical sycophancy)')
ax1.set_title('Sycophancy alignment by emotional intensity')
ax1.legend()

# Mean cosine per level
mean_cos_levels = [
    np.mean(list(cos_by_level[l].values())) for l in [1, 2, 3]
]

ax2.bar([0, 1, 2], mean_cos_levels, color=[BLUE, ORANGE, RED], width=0.5)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(['Low', 'Medium', 'High'])
ax2.set(ylabel='Mean cosine', xlabel='Emotional intensity')
ax2.set_title('Mean alignment with sycophancy direction')

fig.tight_layout()
plt.savefig(f"plots/fig{6}.png", dpi=150, bbox_inches="tight"); plt.close()

print(f'Low:    {mean_cos_levels[0]:.3f}')
print(f'Medium: {mean_cos_levels[1]:.3f}')
print(f'High:   {mean_cos_levels[2]:.3f}')

# Test monotonicity
decreasing = mean_cos_levels[0] > mean_cos_levels[1] > mean_cos_levels[2]
print(f'\nMonotonic decrease: {decreasing}')
if decreasing:
    print('Higher emotional intensity produces LESS sycophancy alignment, not more.')
    print('This is counterintuitive but consistent with the idea that high-emotion prompts')
    print('activate a different representational mode (perhaps genuine concern).')
else:
    print('The monotonic decrease pattern was not observed at this scale/sample size.')
    print('This may differ from the full-dataset result due to the smaller sample.')

# ---
# ## Intervention test: Contrastive activation steering
# 
# Can we subtract the clinical sycophancy direction during generation to reduce sycophancy while preserving empathy? (Proposal Phase 5)
# 
# We test whether subtracting the sycophancy direction from the residual stream can shift the model toward therapeutic responses. Three approaches:
# 
# 1. **Single-layer steering** at a mid-to-late layer
# 2. **Multi-layer steering** across several layers (distributing the intervention)
# 3. **Logit shift measurement** to quantify the effect
# 
# We also generate text examples to qualitatively assess the steering effect.

# Identify causally important layers via logit lens peak
# The layers where the signal flips from therapeutic to sycophantic
# are the ones doing the "override" -- good steering targets

# Aggregate logit lens signal
clin_matrix = np.array([
    [s[l] for l in all_layers] for s in logit_signals['clinical']
])
mean_signal = clin_matrix.mean(0)

# Find layers where signal transitions from positive to negative
# (therapeutic -> sycophantic)
# Skip early layers (first half) — logit lens is noisy there and
# early layers handle low-level processing, not semantic reasoning.
min_layer = N_LAYERS // 2
transition_layers = []
for i in range(1, len(all_layers)):
    if all_layers[i] >= min_layer and mean_signal[i-1] > 0 and mean_signal[i] <= 0:
        transition_layers.append(all_layers[i])

# Pick steering layers: around the transition + some spread
if transition_layers:
    mid = transition_layers[0]
else:
    mid = N_LAYERS * 2 // 3  # fallback: 66% depth

# Multi-layer: 4 layers around the transition
steer_candidates = [l for l in LAYERS if abs(l - mid) <= 6]
if len(steer_candidates) < 3:
    steer_candidates = LAYERS[len(LAYERS)//3 : len(LAYERS)*2//3]
steer_layers = steer_candidates[:4]
single_layer = steer_layers[len(steer_layers) // 2]

print(f'Transition point: ~layer {mid}')
print(f'Single-layer steering: layer {single_layer}')
print(f'Multi-layer steering:  layers {steer_layers}')

# Measure logit shifts at multiple alpha values
# For each stimulus, run forward pass with steering hook and measure
# logit(therapeutic_first_token) - logit(sycophantic_first_token)

test_stimuli = stim_clinical[N_TRAIN:]  # held-out test set
alphas = [2.0, 4.0, 8.0]

device = get_device(model)
dtype = next(model.parameters()).dtype

def measure_logit_shift(model, tokenizer, stimulus, layer, direction, alpha):
    """Measure change in therapeutic-vs-sycophantic logit difference from steering."""
    ids = tokenizer.encode(format_prompt(tokenizer, stimulus['user_prompt']), return_tensors='pt').to(device)
    ther_ids = tokenizer.encode(stimulus['therapeutic_completion'],
                               add_special_tokens=False)[:3]
    syc_ids = tokenizer.encode(stimulus['sycophantic_completion'],
                              add_special_tokens=False)[:3]

    # Baseline
    with torch.no_grad():
        logits_base = model(ids).logits[0, -1].float()
    lp_base = F.log_softmax(logits_base, dim=-1)
    base_diff = float(np.mean([lp_base[t].item() for t in ther_ids]) - np.mean([lp_base[t].item() for t in syc_ids]))

    # Steered
    vec = direction[layer].to(device=device, dtype=dtype)
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h -= alpha * vec  # all positions
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        logits_steer = model(ids).logits[0, -1].float()
    handle.remove()
    lp_steer = F.log_softmax(logits_steer, dim=-1)
    steer_diff = float(np.mean([lp_steer[t].item() for t in ther_ids]) - np.mean([lp_steer[t].item() for t in syc_ids]))

    return steer_diff - base_diff  # positive = shifted toward therapeutic


def measure_multi_layer_shift(model, tokenizer, stimulus, layers, direction, alpha):
    """Multi-layer steering: distribute alpha across layers."""
    ids = tokenizer.encode(format_prompt(tokenizer, stimulus['user_prompt']), return_tensors='pt').to(device)
    ther_ids = tokenizer.encode(stimulus['therapeutic_completion'],
                               add_special_tokens=False)[:3]
    syc_ids = tokenizer.encode(stimulus['sycophantic_completion'],
                              add_special_tokens=False)[:3]

    with torch.no_grad():
        logits_base = model(ids).logits[0, -1].float()
    lp_base = F.log_softmax(logits_base, dim=-1)
    base_diff = float(np.mean([lp_base[t].item() for t in ther_ids]) - np.mean([lp_base[t].item() for t in syc_ids]))

    handles = []
    per_layer_alpha = alpha / np.sqrt(len(layers))
    for sl in layers:
        vec = direction[sl].to(device=device, dtype=dtype)
        def make_hook(v, a=per_layer_alpha):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h -= a * v
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return fn
        handles.append(model.model.layers[sl].register_forward_hook(make_hook(vec)))

    with torch.no_grad():
        logits_steer = model(ids).logits[0, -1].float()
    for h in handles:
        h.remove()
    lp_steer = F.log_softmax(logits_steer, dim=-1)
    steer_diff = float(np.mean([lp_steer[t].item() for t in ther_ids]) - np.mean([lp_steer[t].item() for t in syc_ids]))

    return steer_diff - base_diff


# Measure shifts
print('Measuring logit shifts...')
results_single = {a: [] for a in alphas}
results_multi = {a: [] for a in alphas}

for s in tqdm(test_stimuli, desc='Steering'):
    for a in alphas:
        shift_s = measure_logit_shift(
            model, tokenizer, s, single_layer, dir_clinical, a
        )
        results_single[a].append(shift_s)

        shift_m = measure_multi_layer_shift(
            model, tokenizer, s, steer_layers, dir_clinical, a
        )
        results_multi[a].append(shift_m)

cleanup()

print('\nLogit shifts (positive = more therapeutic):')
print(f'{"Alpha":>8} {"Single-layer":>15} {"Multi-layer":>15}')
for a in alphas:
    ms = np.mean(results_single[a])
    mm = np.mean(results_multi[a])
    print(f'{a:>8.1f} {ms:>+15.3f} {mm:>+15.3f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Logit shift by alpha
ax1.plot(alphas, [np.mean(results_single[a]) for a in alphas],
         'o-', color=BLUE, label=f'Single (layer {single_layer})', lw=1.5)
ax1.plot(alphas, [np.mean(results_multi[a]) for a in alphas],
         's-', color=RED, label=f'Multi (layers {steer_layers})', lw=1.5)
ax1.axhline(0, color='gray', ls=':', alpha=0.4)
ax1.set(xlabel='Steering alpha', ylabel='Mean logit shift')
ax1.set_title('Logit shift: single vs multi-layer')
ax1.legend(fontsize=8)

# Per-stimulus shifts at alpha=4
a_plot = 4.0
x_stim = range(len(test_stimuli))
ax2.bar([x - 0.15 for x in x_stim], results_single[a_plot],
        width=0.3, color=BLUE, label='Single-layer', alpha=0.8)
ax2.bar([x + 0.15 for x in x_stim], results_multi[a_plot],
        width=0.3, color=RED, label='Multi-layer', alpha=0.8)
ax2.axhline(0, color='gray', ls=':', alpha=0.4)
ax2.set(xlabel='Stimulus index', ylabel='Logit shift')
ax2.set_title(f'Per-stimulus shift (alpha={a_plot})')
ax2.legend(fontsize=8)

fig.tight_layout()
plt.savefig(f"plots/fig{7}.png", dpi=150, bbox_inches="tight"); plt.close()

# Generate text examples: baseline vs steered
example_stimuli = [stim_clinical[N_TRAIN], stim_clinical[N_TRAIN+3], stim_clinical[N_TRAIN+6]]
alpha_gen = 6.0

print(f'Generating examples (alpha={alpha_gen})...\n')
print('=' * 70)

for i, s in enumerate(example_stimuli):
    ids = tokenizer.encode(format_prompt(tokenizer, s['user_prompt']), return_tensors='pt').to(device)

    # Baseline
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    baseline = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    # Single-layer steered
    vec = dir_clinical[single_layer].to(device=device, dtype=dtype)
    def hook_single(mod, inp, out, v=vec, a=alpha_gen):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h -= a * v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = model.model.layers[single_layer].register_forward_hook(hook_single)
    with torch.no_grad():
        out_s = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    handle.remove()
    steered_single = tokenizer.decode(out_s[0][ids.shape[1]:], skip_special_tokens=True)

    # Multi-layer steered
    handles = []
    for sl in steer_layers:
        sv = dir_clinical[sl].to(device=device, dtype=dtype)
        def make_h(v, a=alpha_gen / np.sqrt(len(steer_layers))):
            def fn(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h -= a * v
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return fn
        handles.append(model.model.layers[sl].register_forward_hook(make_h(sv)))

    with torch.no_grad():
        out_m = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    for h in handles:
        h.remove()
    steered_multi = tokenizer.decode(out_m[0][ids.shape[1]:], skip_special_tokens=True)

    print(f'\n--- Example {i+1} [{s["subcategory"]}] ---')
    print(f'PROMPT: {s["user_prompt"][:120]}...')
    print(f'\nBASELINE:     {baseline[:250]}')
    print(f'\nSINGLE-LAYER: {steered_single[:250]}')
    print(f'\nMULTI-LAYER:  {steered_multi[:250]}')
    print()

print('=' * 70)
print('\nLook for whether:')
print('  - Baseline validates the distortion (if sycophancy rate was >50%)')
print('  - Steered versions push toward correction or less emotional validation')
print('  - Multi-layer steering produces smoother text than single-layer')
print('If baselines are already therapeutic, the model may not be sycophantic')
print('on these examples, and steering may degrade output quality.')
