(() => {
  'use strict';
  const controls = document.querySelector('.research-controls');
  const cards = [...document.querySelectorAll('.paper-card')];
  const filters = [...document.querySelectorAll('[data-filter]')];
  const search = document.querySelector('#paper-search');
  const count = document.querySelector('#result-number');
  const empty = document.querySelector('#no-results');
  if (!controls || !search || !cards.length) return;
  let active = 'All';
  function update() {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach(card => {
      const match = (active === 'All' || card.dataset.track === active) && card.dataset.search.toLowerCase().includes(query);
      card.hidden = !match;
      if (match) visible++;
    });
    count.textContent = visible;
    empty.hidden = visible > 0;
    filters.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.filter === active)));
  }
  filters.forEach(button => button.addEventListener('click', () => { active = button.dataset.filter; update(); }));
  search.addEventListener('input', update);
  document.querySelectorAll('[data-track-link]').forEach(link => link.addEventListener('click', () => {
    active = link.dataset.trackLink;
    search.value = '';
    update();
  }));
  controls.hidden = false;
})();
