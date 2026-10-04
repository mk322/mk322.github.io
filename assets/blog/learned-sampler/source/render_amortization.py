"""Where computation goes: response-state updates vs shared-parameter updates.

Original schematic; arrows show operations, not measured compute or speedups.
MCMC here denotes the fixed-model preprocessing route in Finetuning with Sampling.
"""
from render_figures import start, text, rect, line, save, PURPLE, MUTED, TEAL, INK


def node(s, x, y, w, label, sub=None, kind='plain', h=56):
    fill, stroke, color = {
        'plain': ('#fff', '#d7dce5', INK),
        'sampler': ('#eee9f7', '#b9abd7', PURPLE),
        'output': ('#edf7f5', '#a4ccc6', TEAL),
    }[kind]
    rect(s, x, y, w, h, fill, stroke, 8)
    text(s, x+w/2, y+(24 if sub else h/2+6), label, 16 if label == 'Current response' else 17, color, 500, 'middle')
    if sub:
        text(s, x+w/2, y+44, sub, 14, MUTED, anchor='middle')


def rule(s, x1, y1, x2, y2):
    s.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="#ded8e9" stroke-dasharray="4 5"/>')


def desktop():
    s = start(800, 486, 'Update a response, or train a reusable sampler?',
              'MCMC repeats propose, score, and accept or reject steps for each prompt, '
              'updating the response while model weights stay fixed. Our method trains '
              'sampler parameters across prompts using sampled responses and target scores. '
              'The trained sampler is then reused for autoregressive generation and verification.')
    rect(s, 12, 12, 776, 202, '#f8f9fb', '#e1e5eb', 12)
    text(s, 32, 44, 'MCMC', 22, weight=600)
    text(s, 124, 44, 'Search for each prompt', 18, MUTED)
    text(s, 766, 43, 'Model weights fixed', 14, MUTED, anchor='end')
    node(s, 32, 80, 142, 'Current response')
    node(s, 216, 80, 150, 'Propose + score')
    node(s, 408, 80, 148, 'Accept / reject')
    node(s, 622, 80, 144, 'Verify', 'SFT response', 'output')
    for a,b in ((176,208),(368,400),(558,614)):
        line(s, f'M{a} 108H{b}')
    line(s, 'M482 139V164H103V139')
    text(s, 292, 190, 'Update the response. Repeat the chain.', 16, MUTED, anchor='middle')
    text(s, 589, 93, 'End', 13, MUTED, anchor='middle')

    rect(s, 12, 230, 776, 244, '#faf8fd', '#ddd5e9', 12)
    text(s, 32, 264, 'Amortized sampler', 22, weight=600)
    rule(s, 492, 282, 492, 454)
    text(s, 32, 297, 'Train across prompts', 17, PURPLE, 500)
    text(s, 532, 297, 'For each new prompt', 17, MUTED, 500)
    node(s, 248, 318, 176, 'Sampler', 'Shared parameters', 'sampler')
    node(s, 248, 404, 176, 'Draw + score')
    line(s, 'M336 377V396', True)
    line(s, 'M246 432H214V346H240', True)
    text(s, 32, 393, 'Update parameters', 17, PURPLE, 500)
    text(s, 32, 416, 'Matching loss', 14, MUTED)
    line(s, 'M427 346H541', True)
    text(s, 484, 333, 'Reuse', 15, PURPLE, anchor='middle')
    node(s, 550, 318, 198, 'Generate', 'Token by token', 'sampler')
    line(s, 'M649 377V396')
    node(s, 550, 404, 198, 'Verify', 'SFT response', 'output')
    save(s, 'amortized-comparison.svg')


def mobile():
    s = start(380, 786, 'Update a response, or train a reusable sampler?',
              'MCMC updates the response in a repeated chain for each prompt. '
              'Amortized training updates shared sampler parameters across prompts. '
              'Data generation reuses the trained sampler, followed by verification.')
    rect(s, 10, 10, 360, 314, '#f8f9fb', '#e1e5eb', 12)
    text(s, 28, 42, 'MCMC', 22, weight=600)
    text(s, 28, 68, 'Search for each prompt', 17, MUTED)
    node(s, 28, 92, 140, 'Current response')
    node(s, 211, 92, 140, 'Propose + score')
    line(s, 'M170 120H203')
    line(s, 'M281 151V183')
    node(s, 211, 191, 140, 'Accept / reject')
    line(s, 'M209 219H98V151')
    text(s, 28, 179, 'Update', 15, MUTED)
    text(s, 28, 198, 'response', 15, MUTED)
    # Route the final state to verification after the search chain.
    line(s, 'M281 250V275H176')
    text(s, 233, 268, 'End', 13, MUTED, anchor='middle')
    node(s, 28, 249, 140, 'Verify', 'SFT response', 'output')
    text(s, 351, 311, 'Model weights fixed', 13, MUTED, anchor='end')

    rect(s, 10, 340, 360, 434, '#faf8fd', '#ddd5e9', 12)
    text(s, 28, 376, 'Amortized sampler', 22, weight=600)
    text(s, 28, 406, 'Train across prompts', 17, PURPLE, 500)
    node(s, 188, 426, 160, 'Sampler', 'Shared parameters', 'sampler')
    node(s, 188, 520, 160, 'Draw + score')
    line(s, 'M300 485V512', True)
    line(s, 'M186 548H158V454H180', True)
    text(s, 28, 478, 'Update', 17, PURPLE, 500)
    text(s, 28, 500, 'parameters', 17, PURPLE, 500)
    text(s, 28, 524, 'Matching loss', 14, MUTED)
    # A separate path carries learned parameters, never MCMC trajectories.
    line(s, 'M350 454H359V687H351', True)
    rule(s, 28, 601, 348, 601)
    text(s, 28, 631, 'For each new prompt', 17, MUTED, 500)
    node(s, 188, 659, 160, 'Generate', 'Token by token', 'sampler')
    text(s, 318, 647, 'Reuse', 14, PURPLE, anchor='middle')
    line(s, 'M186 687H176')
    node(s, 28, 659, 140, 'Verify', 'SFT response', 'output')
    text(s, 28, 749, 'Reuse the policy learned across prompts.', 15, PURPLE)
    save(s, 'amortized-comparison-mobile.svg')


if __name__ == '__main__':
    desktop()
    mobile()
