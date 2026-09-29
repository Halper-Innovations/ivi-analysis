# IVI Analysis — web UI

Vite + React + TypeScript SPA for IVI Analysis. Node is a **build-time
dependency only**: FastAPI serves the built bundle from `webui/dist`, and the
runtime is pure Python.

## Build

```sh
cd webui
npm install
npm run build     # tsc --noEmit && vite build → dist/
```

Then `ivi web` serves the app at http://127.0.0.1:8321.

## Develop

```sh
ivi web           # serves /api on 8321
npm run dev       # Vite dev server on 5173, proxying /api → 8321
```

## Test

```sh
npm run test      # vitest — pure gauge/ladder math
```

Fonts are self-hosted in `public/fonts/` (Besley, IBM Plex Sans and IBM Plex
Mono — OFL-licensed, see `public/fonts/LICENSE.md`) so the UI works offline.
