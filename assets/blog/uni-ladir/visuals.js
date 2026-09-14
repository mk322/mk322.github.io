(() => {
  'use strict';
  const all = (root, selector) => Array.from(root.querySelectorAll(selector));
  const pressed = (buttons, chosen) => buttons.forEach(b => b.setAttribute('aria-pressed', String(b === chosen)));
  document.querySelectorAll('[data-uni]').forEach(root => {
    const status = root.querySelector('[data-status]');
    switch (root.dataset.uni) {
      case 'world': {
        const notes = {
          all: 'A grasp depends on how the object, clearance, and fingers relate. Each observation constrains a different part of that decision.',
          text: 'Language identifies the goal and object relation. A description alone may omit the clearance needed to reach the handle.',
          image: 'Appearance locates the mug and fingers in the view. A 2D overlap does not by itself establish physical contact.',
          geometry: 'Geometry constrains shape, orientation, and clearance. It does not by itself establish whether the gripper has secured the object.',
          state: 'Robot state records configuration. “Gripper closed” and “mug held” are different claims; the latter depends on an object–gripper relation.'
        };
        const buttons = all(root, '[data-view]');
        buttons.forEach(button => button.addEventListener('click', () => {
          pressed(buttons, button);
          all(root, '.uni-view').forEach(view => view.classList.toggle('is-focused', view.classList.contains('uni-' + button.dataset.view)));
          status.textContent = notes[button.dataset.view];
        }));
        break;
      }
      case 'grounding': {
        const buttons = all(root, '[data-objective]');
        const target = root.querySelector('[data-target]');
        buttons.forEach(button => button.addEventListener('click', () => {
          pressed(buttons, button);
          const ground = button.dataset.objective === 'ground';
          target.querySelector('small').textContent = ground ? 'With input and the available latent prefix' : 'From the constructed block';
          target.querySelector('b').textContent = ground ? 'Predict later steps + answer / action' : 'Reconstruct the original teacher representation';
          target.querySelector('span').textContent = ground ? 'Earlier raw teacher steps are hidden.' : 'Retain details that reproduce the source.';
          status.textContent = ground ? 'Keep information because it helps the computation that follows. Grounding trains both the latent constructor and its consumer.' : 'Reconstruction rewards fidelity to the teacher source. That can be useful, but does not directly choose what later reasoning needs.';
        }));
        break;
      }
      case 'results': {
        const buttons = all(root, '[data-result]');
        const select = button => {
          pressed(buttons, button);
          all(root, '[data-chart]').forEach(chart => { chart.hidden = chart.dataset.chart !== button.dataset.result; });
        };
        buttons.forEach(button => button.addEventListener('click', () => select(button)));
        select(buttons[0]);
        break;
      }
      case 'diffusion': {
        let step = 0, sample = 0, timer = null;
        const play = root.querySelector('[data-play]');
        const prev = root.querySelector('[data-prev]');
        const next = root.querySelector('[data-next]');
        const slider = root.querySelector('input[type=range]');
        const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
        const stop = () => {
          if (timer !== null) window.clearInterval(timer);
          timer = null;
          play.textContent = 'Play';
          play.setAttribute('aria-pressed', 'false');
          status.setAttribute('aria-live', 'polite');
        };
        const render = () => {
          const block = Math.floor(step / 5), phase = step % 5, amount = phase / 4;
          const phases = ['noise', 'early denoising', 'denoising', 'late denoising', 'complete'];
          all(root, '[data-block]').forEach((el, i) => {
            const complete = i < block || (i === block && phase === 4);
            el.classList.toggle('is-current', i === block && !complete);
            el.classList.toggle('is-complete', complete);
            el.querySelector('small').textContent = complete ? 'fixed' : i === block ? 'denoising' : 'next';
          });
          all(root, '[data-points] circle').forEach((dot, i) => {
            const startX = 45 + ((i * 83 + sample * 47 + block * 29) % 231);
            const startY = 36 + ((i * 67 + sample * 31 + block * 43) % 144);
            const endX = 90 + (i % 3) * 68 + (sample % 2 ? (i % 2 ? 9 : -9) : 0);
            const endY = 79 + Math.floor(i / 3) * 62 + (sample % 2 ? (i % 3 - 1) * 10 : 0);
            dot.setAttribute('cx', String(startX * (1 - amount) + endX * amount));
            dot.setAttribute('cy', String(startY * (1 - amount) + endY * amount));
          });
          root.querySelector('[data-phase]').textContent = `Block ${block + 1} · ${phases[phase]}`;
          root.querySelector('[data-condition]').textContent = ['Conditioned on the task input x.', 'Conditioned on x and the generated, fixed block z₁.', 'Conditioned on x and the generated, fixed blocks z₁, z₂.'][block];
          root.querySelector('[data-progress]').style.width = `${amount * 100}%`;
          root.querySelector('[data-caption]').textContent = phase === 4 ? (block === 2 ? 'The generated prefix now conditions the answer or action.' : 'This block becomes context for the next one.') : 'All tokens in this block are updated together.';
          status.textContent = phase === 4 ? (block === 2 ? 'Three blocks complete. Generate the answer or action from x and this latent prefix.' : `Block ${block + 1} is fixed. The next block will start from fresh noise.`) : `Denoising block ${block + 1}; ${block} earlier block${block === 1 ? '' : 's'} remain fixed. Sample ${sample + 1}.`;
          slider.value = String(step);
          prev.disabled = step === 0;
          next.disabled = step === 14;
        };
        const seek = value => { stop(); step = Math.max(0, Math.min(14, value)); render(); };
        play.addEventListener('click', () => {
          if (timer !== null) { stop(); return; }
          // Reduced motion keeps all states available via discrete controls.
          if (motion.matches) { seek(step === 14 ? 0 : step + 1); return; }
          if (step === 14) step = 0;
          render();
          play.textContent = 'Pause'; play.setAttribute('aria-pressed', 'true');
          status.setAttribute('aria-live', 'off');
          timer = window.setInterval(() => { step += 1; render(); if (step === 14) stop(); }, 850);
        });
        prev.addEventListener('click', () => seek(step - 1));
        next.addEventListener('click', () => seek(step + 1));
        root.querySelector('[data-reset]').addEventListener('click', () => { sample = 0; seek(0); });
        slider.addEventListener('input', () => seek(Number(slider.value)));
        root.querySelector('[data-sample]').addEventListener('click', () => { sample = (sample + 1) % 4; seek(0); });
        document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
        if (motion.addEventListener) motion.addEventListener('change', () => { stop(); play.hidden = motion.matches; });
        if ('IntersectionObserver' in window) new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stop(); }).observe(root);
        if (motion.matches) play.hidden = true;
        render();
        break;
      }
    }
    all(root, '.uni-controls').forEach(control => { control.hidden = false; });
  });
})();
