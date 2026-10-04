/* Shared per-post page views via https://counterapi.com/#api.
 * Only the production origin records views. Previews use readOnly=true.
 * Counts start when enabled, and represent page views, not unique people. */
(() => {
  'use strict';
  document.querySelectorAll('[data-blog-views]').forEach(root => {
    if (root.dataset.initialized) return;
    root.dataset.initialized = 'true';
    const canonical = new URL(root.dataset.url);
    const count = root.querySelector('[data-view-count]');
    const key = canonical.pathname.replace(/^\/+|\/+$/g, '').replace(/\//g, '--');
    const preview = location.origin !== canonical.origin;
    const endpoint = new URL('https://counterapi.com/api/' +
      encodeURIComponent(canonical.host) + '/view/' + encodeURIComponent(key));
    endpoint.searchParams.set('readOnly', String(preview));

    async function recordView() {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      try {
        const response = await fetch(endpoint.href, {
          signal: controller.signal, credentials: 'omit',
          referrerPolicy: 'no-referrer', cache: 'no-store'
        });
        if (!response.ok) throw new Error('View service unavailable');
        const body = await response.json();
        if (!Number.isSafeInteger(body.value) || body.value < 0) {
          throw new Error('Invalid view count');
        }
        count.textContent = body.value.toLocaleString();
      } catch (_) {
        root.title = 'View count temporarily unavailable';
        count.setAttribute('aria-label', 'View count unavailable');
      } finally {
        clearTimeout(timeout);
      }
    }

    // A background tab records its visit only when the reader opens it.
    if (document.visibilityState === 'visible') recordView();
    else document.addEventListener('visibilitychange', function onVisible() {
      if (document.visibilityState !== 'visible') return;
      document.removeEventListener('visibilitychange', onVisible);
      recordView();
    });
  });
})();
