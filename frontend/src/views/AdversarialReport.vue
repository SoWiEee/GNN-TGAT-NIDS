<template>
  <div>
    <div class="view-header">
      <h2>Adversarial Comparison</h2>
      <button v-if="session.adversarialResult" class="btn-sm" @click="generateReport">
        Export PDF
      </button>
    </div>

    <!-- No flow selected -->
    <div v-if="!session.selectedFlowId" class="empty-state">
      Select a flow from the
      <router-link to="/alerts">Alert List</router-link>
      to generate an adversarial example.
    </div>

    <!-- Loading -->
    <div v-else-if="loading" class="loading">
      Generating adversarial example via C-PGD… (up to 30 s)
    </div>

    <!-- Error -->
    <div v-else-if="error" class="error-msg">{{ error }}</div>

    <!-- Result -->
    <template v-else-if="session.adversarialResult">
      <div class="summary-bar">
        <div class="pill pill-original">
          Original: <strong>{{ result.original.prediction }}</strong>
          {{ (result.original.confidence * 100).toFixed(1) }}%
        </div>
        <span class="arrow">→</span>
        <div class="pill pill-adv">
          Adversarial: <strong>{{ result.adversarial.prediction }}</strong>
        </div>
        <div class="csr" :class="result.adversarial.csr === 1.0 ? 'csr-ok' : 'csr-fail'">
          CSR {{ result.adversarial.csr.toFixed(1) }}
          {{ result.adversarial.csr === 1.0 ? '✅' : '❌' }}
        </div>
      </div>

      <!-- Parameter controls -->
      <div class="params-row">
        <label>
          ε
          <input v-model.number="epsilon" type="number" min="0.01" max="1" step="0.01" />
        </label>
        <label>
          Steps
          <input v-model.number="steps" type="number" min="10" max="200" step="10" />
        </label>
        <button class="btn-regen" @click="generate">Regenerate</button>
      </div>

      <!-- Feature diff table -->
      <table class="diff-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Original</th>
            <th>Adversarial</th>
            <th>Δ%</th>
            <th>Constraint</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="feat in result.adversarial.changed_features"
            :key="feat.name"
            class="changed-row"
          >
            <td class="mono">{{ feat.name }}</td>
            <td>{{ feat.original.toFixed(4) }}</td>
            <td>{{ feat.adversarial.toFixed(4) }}</td>
            <td :class="feat.delta_pct < 0 ? 'neg' : 'pos'">
              {{ feat.delta_pct > 0 ? '+' : '' }}{{ feat.delta_pct.toFixed(1) }}%
            </td>
            <td>{{ feat.constraint_ok ? '✅' : '❌' }}</td>
          </tr>
          <tr v-if="!result.adversarial.changed_features.length">
            <td colspan="5" class="no-change">
              No features changed — adversarial example not found within ε budget. Try increasing ε
              or steps.
            </td>
          </tr>
        </tbody>
      </table>
    </template>

    <!-- Not yet generated -->
    <div v-else class="empty-state">
      <button class="btn-primary" @click="generate">Generate Adversarial Example</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'
import { api } from '@/api'

const session = useSessionStore()
const epsilon = ref(0.1)
const steps = ref(40)
const loading = ref(false)
const error = ref('')

const result = computed(() => session.adversarialResult!)

async function generate() {
  if (!session.selectedFlowId || !session.sessionId) return
  loading.value = true
  error.value = ''
  session.adversarialResult = null
  try {
    await session.generateAdversarial(session.selectedFlowId, epsilon.value, steps.value)
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Generation failed'
  } finally {
    loading.value = false
  }
}

async function generateReport() {
  if (!session.sessionId) return
  const graphPng = sessionStorage.getItem('graph_png_b64') ?? ''
  const { data } = await api.generateReport(session.sessionId, graphPng)
  const apiBase =
    (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'
  window.open(`${apiBase}${data.report_url}?format=pdf`, '_blank')
}

onMounted(() => {
  if (session.selectedFlowId && !session.adversarialResult) generate()
})

watch(
  () => session.selectedFlowId,
  () => {
    session.adversarialResult = null
    error.value = ''
    if (session.selectedFlowId) generate()
  },
)
</script>

<style scoped>
.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
h2 { font-size: 18px; }
.btn-sm {
  padding: 6px 12px;
  background: #334155;
  border: none;
  border-radius: 6px;
  color: #f1f5f9;
  font-size: 13px;
  cursor: pointer;
}
.btn-sm:hover { background: #475569; }
.empty-state { color: #64748b; text-align: center; margin-top: 80px; font-size: 15px; }
.empty-state a { color: #60a5fa; }
.loading { color: #94a3b8; margin-top: 40px; text-align: center; font-size: 14px; }
.error-msg { color: #f87171; margin-top: 20px; font-size: 14px; }

.summary-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}
.pill { padding: 8px 16px; border-radius: 8px; font-size: 14px; }
.pill-original { background: #450a0a; color: #fca5a5; }
.pill-adv { background: #052e16; color: #86efac; }
.arrow { font-size: 20px; color: #64748b; }
.csr { padding: 8px 12px; border-radius: 8px; font-size: 13px; font-weight: 600; }
.csr-ok { background: #052e16; color: #4ade80; }
.csr-fail { background: #450a0a; color: #f87171; }

.params-row {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}
.params-row label {
  font-size: 13px;
  color: #94a3b8;
  display: flex;
  align-items: center;
  gap: 6px;
}
.params-row input {
  width: 72px;
  padding: 4px 8px;
  background: #1e293b;
  border: 1px solid #475569;
  border-radius: 6px;
  color: #f1f5f9;
  font-size: 13px;
}
.btn-regen {
  padding: 6px 14px;
  background: #1d4ed8;
  border: none;
  border-radius: 6px;
  color: #fff;
  font-size: 13px;
  cursor: pointer;
}
.btn-regen:hover { background: #2563eb; }

.diff-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.diff-table th {
  background: #1e293b;
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #334155;
  font-size: 12px;
  color: #94a3b8;
  text-transform: uppercase;
}
.diff-table td { padding: 10px 12px; border-bottom: 1px solid #1e293b44; }
.changed-row td { background: #1a1020; }
.mono { font-family: monospace; font-size: 12px; }
.neg { color: #f87171; font-weight: 600; }
.pos { color: #4ade80; font-weight: 600; }
.no-change { color: #64748b; text-align: center; padding: 24px; }

.btn-primary {
  padding: 12px 28px;
  background: #3b82f6;
  border: none;
  border-radius: 8px;
  color: #fff;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
}
.btn-primary:hover { background: #2563eb; }
</style>
