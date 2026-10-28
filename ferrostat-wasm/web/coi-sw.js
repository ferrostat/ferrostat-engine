// Service worker that makes the page cross-origin isolated, which the
// browser requires before workers may share memory (parallel replications).
// Static hosts such as GitHub Pages cannot send the needed
// headers themselves, so they are added here to every same-origin response.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  if (new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(
    fetch(req).then((res) => {
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      headers.set("Cross-Origin-Embedder-Policy", "require-corp");
      headers.set("Cross-Origin-Resource-Policy", "same-origin");
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    }),
  );
});
