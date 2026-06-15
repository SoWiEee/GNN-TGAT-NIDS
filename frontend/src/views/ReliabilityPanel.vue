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
        <div class="metric-chips">
          <span v-if="metrics.clean_precision != null">P {{ metrics.clean_precision.toFixed(3) }}</span>
          <span v-if="metrics.clean_recall != null">R {{ metrics.clean_recall.toFixed(3) }}</span>
          <span v-if="metrics.clean_macro_f1 != null">Macro {{ metrics.clean_macro_f1.toFixed(3) }}</span>
          <span v-if="metrics.clean_roc_auc != null">AUC {{ metrics.clean_roc_auc.toFixed(3) }}</span>
        </div>

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
          <div v-if="metrics.cpgd_scope" class="metric-note">
            {{ metrics.cpgd_scope === 'full_test' ? 'Full test split' : metrics.cpgd_scope }}
            <span v-if="metrics.cpgd_attack_edges != null">· {{ metrics.cpgd_attack_edges }} attack edges</span>
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

        <details v-if="metrics.per_class && metrics.per_class.length > 0" class="per-class-section">
          <summary class="per-class-toggle">Per-Class Metrics</summary>
          <table class="per-class-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="cls in metrics.per_class" :key="cls.class_id" :class="{ 'low-recall': cls.recall < 0.5 && cls.support > 0 }">
                <td>{{ cls.name }}</td>
                <td>{{ cls.precision.toFixed(3) }}</td>
                <td>{{ cls.recall.toFixed(3) }}</td>
                <td>{{ cls.f1.toFixed(3) }}</td>
                <td>{{ cls.support.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
        </details>

        <details v-if="metrics.confusion_matrix && metrics.class_names" class="per-class-section">
          <summary class="per-class-toggle">Confusion Matrix</summary>
          <div class="cm-wrapper">
            <table class="cm-table">
              <thead>
                <tr>
                  <th></th>
                  <th v-for="name in metrics.class_names" :key="name" class="cm-header">{{ name.slice(0, 5) }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, ri) in metrics.confusion_matrix" :key="ri">
                  <td class="cm-row-label">{{ (metrics.class_names ?? [])[ri]?.slice(0, 5) }}</td>
                  <td
                    v-for="(val, ci) in row"
                    :key="ci"
                    class="cm-cell"
                    :style="{ background: cmColor(val, metrics.confusion_matrix!) }"
                  >{{ val > 0 ? val : '' }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const loading = ref(true)

function cmColor(val: number, matrix: number[][]): string {
  if (val === 0) return 'transparent'
  const maxVal = Math.max(...matrix.flat())
  const intensity = Math.min(val / Math.max(maxVal, 1), 1)
  return `rgba(59, 130, 246, ${0.15 + intensity * 0.7})`
}

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
.metric-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: -10px 0 18px; }
.metric-chips span {
  color: #cbd5e1;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 11px;
}
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
.metric-note { color: #94a3b8; font-size: 11px; margin-top: 4px; }
.per-class-section { margin-top: 12px; }
.per-class-toggle { cursor: pointer; font-size: 12px; color: #60a5fa; user-select: none; }
.per-class-toggle:hover { text-decoration: underline; }
.per-class-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 11px; }
.per-class-table th { text-align: left; color: #94a3b8; border-bottom: 1px solid #334155; padding: 4px 6px; }
.per-class-table td { padding: 4px 6px; border-bottom: 1px solid #1e293b; }
.low-recall { color: #f87171; }
.cm-wrapper { overflow-x: auto; margin-top: 8px; }
.cm-table { border-collapse: collapse; font-size: 10px; }
.cm-table th, .cm-table td { padding: 3px 5px; text-align: center; min-width: 36px; }
.cm-header { color: #94a3b8; writing-mode: vertical-lr; transform: rotate(180deg); font-weight: 400; }
.cm-row-label { text-align: right; color: #94a3b8; font-weight: 400; }
.cm-cell { border: 1px solid #1e293b; font-variant-numeric: tabular-nums; }
</style>
