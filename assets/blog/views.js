/* Shared page views, using Vercount's public event endpoint:
 * https://github.com/EvanNotFound/vercount
 * Record one visible production-page load. No local or preview visits are sent.
 * The canonical article URL keeps query strings and fragments in one counter.
 */
(() => {
  'use strict';
  document.querySelectorAll('[data-blog-views]').forEach(root => {
    if (root.dataset.initialized) return;
    root.dataset.initialized = 'true';
    const canonical = new URL(root.dataset.url);
    canonical.search = ''; canonical.hash = '';
    const count = root.querySelector('[data-view-count]');
    if (location.origin !== canonical.origin) {
      root.title = 'Page views are recorded on the published website';
      return;
    }
    async function recordView() {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      try {
        const response = await fetch('https://events.vercount.one/api/v2/log', {
          method:'POST', headers:{'Content-Type':'application/json'},
          // Only page views are displayed; no persistent visitor ID is needed.
          body:JSON.stringify({url:canonical.href,isNewUv:false}),
          signal:controller.signal, credentials:'omit', referrerPolicy:'no-referrer'
        });
        if (!response.ok) throw new Error('View service unavailable');
        const body = await response.json();
        const value = body.data && body.data.page_pv;
        if (body.status !== 'success' || !Number.isSafeInteger(value) || value < 0) {
          throw new Error('Invalid view count');
        }
        count.textContent = value.toLocaleString();
      } catch (_) {
        root.title = 'View count temporarily unavailable';
        count.setAttribute('aria-label','View count unavailable');
      } finally { clearTimeout(timeout); }
    }
    if (document.visibilityState === 'visible') recordView();
    else document.addEventListener('visibilitychange', function onVisible() {
      if (document.visibilityState !== 'visible') return;
      document.removeEventListener('visibilitychange', onVisible);
      recordView();
    });
  });
})();
