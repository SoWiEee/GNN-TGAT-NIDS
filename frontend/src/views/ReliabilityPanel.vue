<template>
  <div>
    <h2 class="page-title">Model Reliability</h2>
    <p class="subtitle">
      Pre-computed on NF-UNSW-NB15-v2 test split. Shows how trustworthy the system is under
      adversarial conditions.
    </p>

    <div v-if="loading" class="loading">Loading metrics…</div>

    <div v-else-if="!session.reliability" class="empty-state">
      Metrics not available. Run
      <code>uv run python scripts/compute_reliability_metrics.py</code> after training.
    </div>

    <div v-else class="model-grid">
      <div
        v-for="(metrics, modelName) in session.reliability"
        :key="modelName"
        class="model-card"
      >
        <h3>{{ String(modelName).toUpperCase() }}</h3>

        <div class="metric">
          <div class="metric-label">Clean F1</div>
          <div class="metric-bar">
            <div
              class="metric-fill"
              :style="{ width: (metrics.clean_f1 ?? 0) * 100 + '%', background: '#22c55e' }"
            />
          </div>
          <div class="metric-value">
            {{ metrics.clean_f1 != null ? metrics.clean_f1.toFixed(3) : 'TBD' }}
          </div>
        </div>

        <div class="metric">
          <div class="metric-label">Detection Rate under C-PGD (ε=0.1)</div>
          <div class="metric-bar">
            <div
              class="metric-fill"
              :style="{
                width: (metrics.dr_under_cpgd_eps01 ?? 0) * 100 + '%',
                background: '#f97316',
              }"
            />
          </div>
          <div class="metric-value">
            {{
              metrics.dr_under_cpgd_eps01 != null
                ? metrics.dr_under_cpgd_eps01.toFixed(3)
                : 'TBD'
            }}
          </div>
        </div>

        <div class="metric">
          <div class="metric-label">ΔF1 after Adversarial Training</div>
          <div class="metric-bar">
            <div
              class="metric-fill"
              :style="{
                width: Math.min(Math.abs(metrics.delta_f1_after_adv_training ?? 0) * 500, 100) + '%',
                background: '#3b82f6',
              }"
            />
          </div>
          <div class="metric-value">
            {{
              metrics.delta_f1_after_adv_training != null
                ? '+' + metrics.delta_f1_after_adv_training.toFixed(3)
                : 'TBD'
            }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const loading = ref(true)

onMounted(async () => {
  await session.loadReliability()
  loading.value = false
})
</script>

<style scoped>
.page-title { font-size: 18px; margin-bottom: 8px; }
.subtitle { font-size: 13px; color: #64748b; margin-bottom: 28px; }
.loading { color: #64748b; }
.empty-state { color: #64748b; font-size: 14px; }
.empty-state code {
  background: #1e293b;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}
.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 20px;
}
.model-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 24px;
}
.model-card h3 { font-size: 15px; margin-bottom: 20px; color: #60a5fa; }
.metric { margin-bottom: 18px; }
.metric-label { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
.metric-bar {
  height: 8px;
  background: #334155;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 4px;
}
.metric-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
.metric-value { font-size: 20px; font-weight: 700; }
</style>
