(() => {
  'use strict';
  document.querySelectorAll('[data-walkthrough]').forEach(root => {
    const buttons = [...root.querySelectorAll('[data-stage]')];
    const box = root.querySelector('.uni-stage-highlight');
    const explanation = root.querySelector('[data-explanation]');
    const play = root.querySelector('[data-play]');
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    const full = explanation.innerHTML;
    const stages = [
      {top:0,height:41,text:'Construct: modality encoders produce teacher features sᵢ. Learnable queries q read only their own step through the shared backbone. Their hidden states form clean latent blocks zᵢ*. The mask at right makes this locality explicit.'},
      {top:41,height:20.5,text:'Ground: the task input x and available clean latent prefix predict later teacher steps and the final output y. Earlier raw teacher steps are hidden. This supervision trains the constructor and consumer to retain useful information.'},
      {top:61.5,height:19.5,text:'Learn diffusion: perturb the current clean block with noise and predict a denoising velocity, conditioned on x and the clean prefix. Snowflakes mark detached targets and conditioning blocks. The diffusion loss still updates the shared backbone.'},
      {top:81,height:9.5,text:'Inference: begin with x, generate z₁, then generate subsequent blocks conditioned on the generated prefix. Each new block is denoised before moving on. The completed sequence conditions y; no teacher traces are required or decoded.'}
    ];
    let current=-1,timer=null;
    function stop(){if(timer!==null)window.clearInterval(timer);timer=null;play.textContent='Play walkthrough';play.setAttribute('aria-pressed','false');explanation.setAttribute('aria-live','polite');}
    function show(index){current=index;buttons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.stage===(index<0?'all':String(index)))));box.hidden=index<0;if(index<0){explanation.innerHTML=full;return;}const s=stages[index];box.style.top=s.top+'%';box.style.height=s.height+'%';explanation.textContent=s.text;}
    buttons.forEach(b=>b.addEventListener('click',()=>{stop();show(b.dataset.stage==='all'?-1:Number(b.dataset.stage));}));
    play.addEventListener('click',()=>{if(timer!==null){stop();return;}if(reduced.matches)return;show(0);play.textContent='Pause';play.setAttribute('aria-pressed','true');explanation.setAttribute('aria-live','off');timer=window.setInterval(()=>{show(current+1);if(current===3)stop();},5000);});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
    if('IntersectionObserver'in window)new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();}).observe(root);
    const motion=()=>{stop();play.hidden=reduced.matches;};if(reduced.addEventListener)reduced.addEventListener('change',motion);
    motion();root.querySelector('.uni-walk-controls').hidden=false;
  });
})();
