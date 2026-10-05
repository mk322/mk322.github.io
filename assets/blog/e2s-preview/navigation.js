/* Keep the small-screen contents collapsible; expose it in the wide margin. */
(function () {
  const contents = document.querySelector('.sampler-toc details');
  const wide = window.matchMedia('(min-width: 1320px)');
  function sync() { if (contents) contents.open = wide.matches; }
  sync();
  wide.addEventListener('change', sync);
})();
