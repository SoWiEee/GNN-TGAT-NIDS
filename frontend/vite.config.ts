import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-cytoscape': ['cytoscape'],
          'vendor-plotly': ['plotly.js-basic-dist-min'],
          'vendor-vue': ['vue', 'pinia', 'vue-router'],
        },
      },
    },
  },
})
