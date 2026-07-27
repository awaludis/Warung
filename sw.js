/* Service worker Kasir Kelontong
   Strategi:
   - App shell (index.html, manifest) di-cache saat install.
   - Skrip Supabase dari CDN & aset lain: cache-saat-berhasil (runtime).
   - Navigasi: coba jaringan dulu, jatuh ke cache saat offline.
   Catatan: panggilan ke Supabase (REST/Auth) SELALU lewat jaringan,
   tidak pernah di-cache — data transaksi ditangani oleh outbox di app.
*/
const CACHE = 'kasir-kelontong-v1';
const SHELL = ['./', './index.html', './manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // Jangan pernah cache panggilan ke Supabase (data harus segar).
  if (/supabase\.(co|in)$/.test(url.hostname) || url.hostname.includes('supabase')) {
    return; // biarkan lewat jaringan seperti biasa
  }

  // Navigasi halaman: network-first → fallback index.html dari cache.
  if (req.mode === 'navigate') {
    e.respondWith(
      fetch(req).catch(() => caches.match('./index.html'))
    );
    return;
  }

  // Aset lain (termasuk CDN Supabase JS): cache-first, isi cache saat berhasil.
  e.respondWith(
    caches.match(req).then(hit => {
      if (hit) return hit;
      return fetch(req).then(res => {
        if (res && res.status === 200 &&
            (url.origin === location.origin || res.type === 'cors' || res.type === 'basic')) {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(req, clone));
        }
        return res;
      }).catch(() => caches.match('./index.html'));
    })
  );
});
