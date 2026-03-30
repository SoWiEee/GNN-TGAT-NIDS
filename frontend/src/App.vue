<template>
  <div class="app">
    <nav class="navbar">
      <span class="brand">GNN-NIDS Analyzer</span>
      <div class="nav-links">
        <router-link to="/">Upload</router-link>
        <template v-if="session.sessionId">
          <router-link to="/graph">Traffic Graph</router-link>
          <router-link to="/alerts">Alerts</router-link>
          <router-link to="/timeline">Timeline</router-link>
          <router-link to="/reliability">Reliability</router-link>
          <router-link to="/adversarial">Adversarial</router-link>
        </template>
      </div>
    </nav>
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import { useSessionStore } from '@/stores/session'
const session = useSessionStore()
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #f1f5f9; }
.app { min-height: 100vh; display: flex; flex-direction: column; }
.navbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px; height: 56px; background: #1e293b; border-bottom: 1px solid #334155;
}
.brand { font-weight: 700; font-size: 16px; color: #60a5fa; }
.nav-links { display: flex; gap: 20px; }
.nav-links a { color: #94a3b8; text-decoration: none; font-size: 14px; }
.nav-links a.router-link-active { color: #f1f5f9; font-weight: 600; }
.main-content { flex: 1; padding: 24px; max-width: 1400px; margin: 0 auto; width: 100%; }
</style>
