(() => {
  'use strict';
  const nav = document.querySelector('[data-uni-toc]');
  if (!nav) return;
  const details = nav.querySelector('details');
  const desktop = window.matchMedia('(min-width: 1280px)');
  const adapt = () => { details.open = desktop.matches; };
  adapt();
  desktop.addEventListener('change', adapt);
  const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
  const sections = links.map(link => document.getElementById(link.hash.slice(1)));
  function highlight() {
    let current = 0;
    sections.forEach((section, i) => {
      if (section && section.getBoundingClientRect().top <= 140) current = i;
    });
    links.forEach((link, i) => {
      if (i === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  let queued = false;
  window.addEventListener('scroll', () => {
    if (queued) return;
    queued = true;
    window.requestAnimationFrame(() => { highlight(); queued = false; });
  }, { passive: true });
  window.addEventListener('resize', highlight);
  links.forEach(link => link.addEventListener('click', event => {
    const section = document.getElementById(link.hash.slice(1));
    if (!section) return;
    // The theme's delegated smooth-scroll handler otherwise suppresses the URL
    // fragment and scrolls without accounting for the collapsed mobile menu.
    event.preventDefault();
    event.stopPropagation();
    if (!desktop.matches) details.open = false;
    window.history.pushState(null, '', link.hash);
    section.scrollIntoView({ behavior: 'instant', block: 'start' });
    section.setAttribute('tabindex', '-1');
    section.focus({ preventScroll: true });
    highlight();
  }));
  highlight();
})();
