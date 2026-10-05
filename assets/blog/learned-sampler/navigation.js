/* Keep native anchor semantics while bypassing the theme's jQuery scroll animation. */
document.addEventListener('click', function (event) {
  const link = event.target.closest('.sampler-toc a[href^="#"]');
  if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  const target = document.getElementById(link.hash.slice(1));
  if (!target) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  target.scrollIntoView({ behavior: 'instant', block: 'start' });
  history.pushState(null, '', link.hash);
}, true);

/* Highlight the section currently being read without replacing native links. */
(function () {
  const links = Array.from(document.querySelectorAll('.sampler-toc a[href^="#"]'));
  const sections = links.map(link => ({ link, section: document.getElementById(link.hash.slice(1)) })).filter(item => item.section);
  let queued = false;
  function update() {
    queued = false;
    let active = sections[0];
    for (const item of sections) {
      if (item.section.getBoundingClientRect().top <= 90) active = item;
    }
    for (const item of sections) {
      if (item === active) item.link.setAttribute('aria-current', 'location');
      else item.link.removeAttribute('aria-current');
    }
  }
  window.addEventListener('scroll', function () {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, { passive: true });
  update();
}());
