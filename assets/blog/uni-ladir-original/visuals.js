(() => {
  'use strict';
  document.querySelectorAll('[data-fig2-animation]').forEach(root => {
    const select = root.querySelector('[data-fig2-select]');
    const play = root.querySelector('[data-fig2-play]');
    const previous = root.querySelector('[data-fig2-prev]');
    const next = root.querySelector('[data-fig2-next]');
    const zoom = root.querySelector('[data-fig2-zoom]');
    const svg = root.querySelector('svg');
    const viewport = root.querySelector('.uni-fig2-viewport');
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const full = [0,0,1800,1215];
    const encoding = [15,15,950,530], continuation = [980,105,805,375];
    const diffusion = [585,810,1200,270], inference = [15,1100,1770,105];
    const chain = [20,555,1760,240], encodePath = [20,615,735,110];
    const contPath = [745,560,695,88], diffPath = [785,700,655,90];
    const stages = [
      {title:'One backbone learns the thoughts and how to generate them.', text:'Teacher steps supply training examples of reasoning. The shared backbone encodes them into thought blocks, learns to use the blocks for later predictions, and learns to generate them from noise. At inference, only the task input is needed. Solid arrows carry information; dashed arrows show gradients; snowflakes stop gradients.', regions:[full], crop:full, gradients:['continuation','diffusion']},
      {title:'1. Latent Encoding: turn each teacher step into a thought.', text:'A teacher step first becomes features s. Learnable queries q collect information from those features through the shared backbone, producing a thought block z★. The attention mask keeps each query block local to its own step. Text, images, 3D point clouds, and robot states use the same latent interface.', regions:[encoding,encodePath], crop:[5,5,965,545], gradients:[]},
      {title:'2. Continuation Prediction: learn what a thought needs to keep.', text:'The task input x and earlier thought blocks z★ help predict later teacher steps s and the final output y. Lcont measures those prediction errors, including final-output supervision. To help these predictions, each thought must preserve useful information from its teacher step.', regions:[continuation,encodePath,contPath,[730,590,90,88]], crop:[975,95,815,390], gradients:[]},
      {title:'3. Diffusion Training: learn to generate the next thought.', text:'Add noise to a clean thought block from the encoder. Given the task input x and earlier blocks, the reasoner learns to denoise it with the diffusion loss Ldiff. Snowflakes mark the clean target and prefix as fixed values for this loss. The original figure shows the denoised-block view; training uses flow matching.', regions:[diffusion,diffPath], crop:[580,805,1210,280], gradients:[]},
      {title:'4. Inference: produce thoughts without a teacher.', text:'Start with the task input x. Generate the first thought block from noise, then use the input and previous thoughts to generate the next one. After the chain ends, predict the answer or robot action y. The teacher steps used during training are no longer needed.', regions:[inference], crop:[10,1100,1780,110], gradients:[]},
      {title:'5. Continuation gradients teach the encoder what matters.', text:'Follow the blue dashed arrows from Lcont back through Continuation Prediction, the thought block z★, and Latent Encoding. This path updates both the predictor and the encoder: a teacher step is encoded according to how its information helps later reasoning and the final output.', regions:[continuation,encodePath,contPath,[580,590,270,110]], crop:[325,555,1120,235], gradients:['continuation']},
      {title:'6. Diffusion gradients train the generator; the targets stay fixed.', text:'The orange dashed arrow runs from Ldiff into Diffusion Training. It does not continue through the snowflake into the clean thought blocks. This prevents the diffusion loss from changing its targets through that path. It still updates the backbone weights shared with the encoder.', regions:[diffusion,diffPath,[825,715,80,65]], crop:[785,690,665,110], gradients:['diffusion']},
      {title:'7. Joint Training: combine both losses in one shared update.', text:'All three components use the same backbone weights. Continuation supervision trains thoughts that help later reasoning; diffusion supervision trains the model to generate those thoughts. Their combined gradients update the shared weights, so the thought space and its generator learn together.', regions:[chain], crop:[15,550,1775,250], gradients:['continuation','diffusion']}
    ];
    let index = 0, timer = null, autoStarted = false;
    zoom.checked = window.matchMedia('(max-width: 600px)').matches;
    function stop() {
      if (timer !== null) window.clearInterval(timer);
      timer = null; root.dataset.running = 'false';
      play.textContent = 'Play'; play.setAttribute('aria-pressed','false');
      root.querySelector('[data-fig2-note]').setAttribute('aria-live','polite');
    }
    function show(value) {
      index = value; select.value = String(index); const stage = stages[index];
      root.dataset.stage = String(index);
      root.querySelector('[data-fig2-clip]').setAttribute('d',stage.regions.map(([x,y,w,h]) => `M${x} ${y}h${w}v${h}h${-w}Z`).join(''));
      root.querySelectorAll('[data-fig2-gradient]').forEach(el => { el.style.display = stage.gradients.includes(el.dataset.fig2Gradient) ? '' : 'none'; });
      root.querySelector('.uni-fig2-stop').toggleAttribute('hidden',index !== 6);
      const crop = zoom.checked && index !== 0 ? stage.crop : full;
      svg.setAttribute('viewBox',crop.join(' '));
      root.dataset.zoomed = String(zoom.checked && index !== 0);
      root.querySelector('[data-fig2-heading]').textContent = stage.title;
      root.querySelector('[data-fig2-explanation]').textContent = stage.text;
      previous.disabled = index === 0; next.disabled = index === stages.length - 1;
      viewport.scrollLeft = 0;
    }
    function start() {
      if (reduced.matches || timer !== null) return;
      if (index === 0 || index === stages.length-1) show(1);
      root.dataset.running='true'; play.textContent='Pause'; play.setAttribute('aria-pressed','true');
      root.querySelector('[data-fig2-note]').setAttribute('aria-live','off');
      timer=window.setInterval(() => {show(index+1);if(index === stages.length-1) stop();},8000);
    }
    function manual(action) { autoStarted = true; stop(); action(); }
    select.addEventListener('change', () => manual(() => show(Number(select.value))));
    previous.addEventListener('click', () => manual(() => show(Math.max(0,index-1))));
    next.addEventListener('click', () => manual(() => show(Math.min(stages.length-1,index+1))));
    root.querySelector('[data-fig2-reset]').addEventListener('click', () => manual(() => show(0)));
    zoom.addEventListener('change', () => manual(() => show(index)));
    play.addEventListener('click', () => {autoStarted=true;if(timer !== null) stop();else start();});
    document.addEventListener('visibilitychange', () => {if(document.hidden) stop();});
    if ('IntersectionObserver' in window) new IntersectionObserver(entries => {
      const visible = entries[0].isIntersecting;
      if (!visible) stop();
      else if (!autoStarted && !reduced.matches && !document.hidden) {autoStarted=true;start();}
    },{threshold:.45}).observe(viewport);
    const motion = () => {stop();play.hidden=reduced.matches;};
    reduced.addEventListener('change',motion);
    root.classList.add('is-interactive'); root.querySelector('.uni-fig2-controls').hidden=false;
    motion();show(0);
  });
})();

(() => {
  document.querySelectorAll('[data-uni-citation]').forEach(root => {
    const button = root.querySelector('[data-copy-citation]');
    const code = root.querySelector('[data-citation-text]');
    const status = root.querySelector('[data-citation-status]');
    button.hidden = false;
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent.trim() + '\n');
        button.textContent = 'Copied';
        status.textContent = 'BibTeX copied to clipboard.';
      } catch (_) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(code);
        selection.removeAllRanges(); selection.addRange(range);
        status.textContent = 'Citation selected. Press Command+C or Control+C to copy.';
      }
    });
  });
})();
