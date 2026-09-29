/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Dev-time only: the built bundle is served by FastAPI itself.
      "/api": "http://127.0.0.1:8321",
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
