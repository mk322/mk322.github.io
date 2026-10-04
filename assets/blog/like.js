/* Anonymous shared counts using the community Applause backend recommended at
 * https://github.com/ColinEberhardt/applause-button . No account or bundled SDK.
 * A remembered vote is per browser, not a verified unique person. */
(() => {
  'use strict';
  const api = 'https://applause.chabouis.fr';
  document.querySelectorAll('[data-blog-like]').forEach(root => {
    const url = root.dataset.url;
    const key = 'blog-like:' + url;
    const button = root.querySelector('button');
    const label = root.querySelector('[data-like-label]');
    const count = root.querySelector('[data-like-count]');
    const status = root.querySelector('[data-like-status]');
    const endpoint = path => api + path + '?url=' + encodeURIComponent(url);
    const markLiked = () => {
      button.setAttribute('aria-pressed', 'true');
      button.disabled = true;
      label.textContent = 'Liked';
      status.textContent = 'Thank you!';
    };
    const isPreview = location.origin !== new URL(url).origin;
    let liked = false;
    try { liked = localStorage.getItem(key) === '1'; } catch (_) {}
    if (liked) markLiked();
    async function request(path, options = {}) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      try {
        const response = await fetch(endpoint(path), { ...options, signal: controller.signal,
          credentials: 'omit', referrerPolicy: 'strict-origin-when-cross-origin' });
        if (!response.ok) throw new Error('Like service unavailable');
        const body = (await response.text()).trim();
        if (!/^\d+$/.test(body)) throw new Error('Invalid like count');
        const total = Number(body);
        if (!Number.isSafeInteger(total)) throw new Error('Invalid like count');
        return total;
      } finally { clearTimeout(timeout); }
    }
    let initial = request('/get-claps').then(total => {
      count.textContent = total.toLocaleString();
    }).catch(() => {
      if (!liked && !isPreview) status.textContent = 'Count unavailable · you can still try liking';
    });
    if (isPreview) {
      button.disabled = true;
      status.textContent = 'Preview';
      return;
    }
    button.addEventListener('click', async () => {
      if (liked) return;
      button.disabled = true;
      status.textContent = 'Saving…';
      await initial; // Keep a late initial count from overwriting the saved total.
      try {
        const total = await request('/update-claps', {
          method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: JSON.stringify('1,4.0.7')
        });
        count.textContent = total.toLocaleString();
        liked = true;
        try { localStorage.setItem(key, '1'); } catch (_) {}
        markLiked();
      } catch (_) {
        button.disabled = false;
        status.textContent = 'Could not save · please try again later';
      }
    });
  });
})();
