const CACHE_NAME = 'Bone-Fracture-app-v1';
const FILES_TO_CACHE = [
  '/',
  'manifest.json',
  'icons/bone_fracture.jpg',
  'icons/bone_Fracture192.jpg'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(FILES_TO_CACHE))
      .catch(err => console.warn('Cache failed:', err))
  );
  self.skipWaiting();
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(resp => resp || fetch(event.request))
  );
});

