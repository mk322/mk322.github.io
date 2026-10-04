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
