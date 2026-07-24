const CACHE = "atcroster-shell-v1";
const SHELL = ["/static/styles.css", "/static/favicon.svg", "/static/manifest.webmanifest"];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin || !url.pathname.startsWith("/static/")) return;
  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});
