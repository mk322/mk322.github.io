(() => {
  'use strict';
  document.querySelectorAll('[data-training-flow]').forEach(root => {
    const select = root.querySelector('[data-flow-select]');
    const play = root.querySelector('[data-flow-play]');
    const previous = root.querySelector('[data-flow-prev]');
    const next = root.querySelector('[data-flow-next]');
    const explanation = root.querySelector('[data-flow-explanation]');
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const stages = [
      {nodes:null, edges:null, text:'One backbone, two losses. Continuation gradients pass through thought blocks into the encoder. Diffusion gradients train the velocity predictor and stop at detached blocks. Both losses update the shared weights θ.'},
      {nodes:['teacher','encoder','thought'], edges:['encode'], text:'1. Encode locally. The shared encoder maps each teacher step to a clean thought block. Each step is encoded on its own; the task input enters the prediction paths.'},
      {nodes:['thought','task','continuation','contloss'], edges:['cont-forward'], text:'2. Continuation forward. The task input and permitted thought prefix predict later teacher steps and the final output. Comparing predictions with their targets gives Lcont, including final-output supervision.'},
      {nodes:['contloss','continuation','thought','encoder'], edges:['cont-backward'], text:'3. Continuation backward. Gradients run from the loss through the continuation predictor and thought blocks into the encoder. They train the encoded blocks to retain information useful for later reasoning.'},
      {nodes:['thought','stop','noise','noisy','prefix','diffusion','velocity','diffloss'], edges:['detach','diff-forward'], text:'4. Diffusion forward. Detach the clean target and earlier blocks. Mix the target with Gaussian noise at time t. The backbone predicts a velocity; Ldiff compares it with the fixed noise-to-target velocity.'},
      {nodes:['diffusion','diffloss','noisy','prefix','stop','stop-bar','velocity'], edges:['diff-backward','diff-stop'], text:'5. Diffusion backward. Ldiff updates the velocity predictor. Stop-gradient blocks its route into the clean target and conditioning prefix. The encoder still changes when the shared backbone weights are updated.'},
      {nodes:['encoder','continuation','diffusion'], edges:[], text:'6. One shared update. Combine Lcont + λ Ldiff and update θ. The encoder, continuation predictor, and diffusion predictor use the same backbone weights. These are two gradient contributions to joint training, not two independently trained systems.'}
    ];
    let index = 0, timer = null;
    function stop() {
      if (timer !== null) window.clearInterval(timer);
      timer = null; root.dataset.running = 'false';
      play.textContent = 'Play'; play.setAttribute('aria-pressed','false');
      explanation.setAttribute('aria-live','polite');
    }
    function show(value) {
      index = value; select.value = String(index); const stage = stages[index];
      root.querySelectorAll('[data-node]').forEach(el => el.classList.toggle('is-active', stage.nodes === null || stage.nodes.includes(el.dataset.node)));
      root.querySelectorAll('[data-edge]').forEach(el => el.classList.toggle('is-active', stage.edges === null || stage.edges.includes(el.dataset.edge)));
      explanation.textContent = stage.text; previous.disabled = index === 0; next.disabled = index === stages.length - 1;
    }
    select.addEventListener('change', () => {stop();show(Number(select.value));});
    previous.addEventListener('click', () => {stop();show(Math.max(0,index-1));});
    next.addEventListener('click', () => {stop();show(Math.min(stages.length-1,index+1));});
    root.querySelector('[data-flow-reset]').addEventListener('click', () => {stop();show(0);});
    play.addEventListener('click', () => {
      if (timer !== null) {stop();return;}
      if (reduced.matches) return;
      if (index === 0 || index === stages.length-1) show(1);
      root.dataset.running='true'; play.textContent='Pause'; play.setAttribute('aria-pressed','true'); explanation.setAttribute('aria-live','off');
      timer=window.setInterval(() => {show(index+1);if(index === stages.length-1) stop();},4500);
    });
    document.addEventListener('visibilitychange', () => {if(document.hidden) stop();});
    if ('IntersectionObserver' in window) new IntersectionObserver(entries => {if(!entries[0].isIntersecting) stop();}).observe(root);
    const motion = () => {stop();play.hidden=reduced.matches;};
    reduced.addEventListener('change',motion);
    root.classList.add('is-interactive'); root.querySelector('.uni-training-controls').hidden=false;
    motion();show(0);
  });
})();
