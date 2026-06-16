<template>
  <div class="view-header">
    <h2>Explainability</h2>
    <div class="controls">
      <select v-model="model">
        <option value="graphsage">GraphSAGE</option>
        <option value="gat">GAT</option>
        <option value="egraphsage">E-GraphSAGE</option>
        <option value="tgat">TGAT</option>
        <option value="tgn">TGN</option>
      </select>
      <div class="top-k-group">
        <label>Top-K</label>
        <input v-model.number="topK" type="number" min="1" max="20" />
      </div>
      <button class="btn-explain" :disabled="session.explainLoading" @click="runTopK">
        <span v-if="session.explainLoading">Explaining...</span>
        <span v-else>Explain Top Alerts</span>
      </button>
    </div>
  </div>

  <div v-if="session.explainLoading" class="loading">
    <div class="spinner" />
    <p>Running {{ isTemporal ? 'gradient attribution' : 'GNNExplainer' }}...</p>
  </div>

  <div v-else-if="session.explainResults.length === 0" class="empty-state">
    <p>Select a model and click "Explain Top Alerts" to see which features drive the model's attack predictions.</p>
    <p class="hint">Temporal models (TGAT/TGN) use integrated gradients. Static models (GraphSAGE/GAT/E-GraphSAGE) use GNNExplainer.</p>
  </div>

  <div v-else class="results">
    <div
      v-for="(result, idx) in session.explainResults"
      :key="result.edge_idx"
      class="explain-card"
    >
      <div class="card-header">
        <span class="rank" v-if="result.rank">#{{ result.rank }}</span>
        <span class="edge-id">Edge {{ result.edge_idx }}</span>
        <span class="nodes">
          {{ result.src_node ?? result.src }} &rarr; {{ result.dst_node ?? result.dst }}
        </span>
        <span v-if="result.timestamp != null" class="timestamp">
          t={{ result.timestamp.toFixed(1) }}s
        </span>
        <span class="class-badge" :class="result.predicted_class > 0 ? 'attack' : 'benign'">
          Class {{ result.predicted_class }}
        </span>
        <span class="confidence">{{ (result.confidence * 100).toFixed(1) }}%</span>
        <span v-if="result.method" class="method-tag">{{ result.method }}</span>
      </div>

      <div class="feature-bars">
        <div class="bar-label-row">
          <span class="bar-title">Feature Importance</span>
        </div>
        <div
          v-for="feat in result.top_features"
          :key="feat.name"
          class="bar-row"
        >
          <span class="feat-name">{{ feat.name }}</span>
          <div class="bar-track">
            <div
              class="bar-fill"
              :style="{ width: barWidth(feat, result) }"
              :class="feat.attribution != null && feat.attribution < 0 ? 'negative' : 'positive'"
            />
          </div>
          <span class="feat-value">{{ feat.importance.toFixed(4) }}</span>
        </div>
      </div>

      <div v-if="result.edge_self_importance != null" class="meta-row">
        Edge self-importance: <strong>{{ result.edge_self_importance.toFixed(4) }}</strong>
      </div>
      <div v-if="result.window != null" class="meta-row">
        Window: <strong>{{ result.window }}</strong>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useSessionStore } from '@/stores/session'
import type { ExplainResult, ExplainFeature } from '@/api'

const session = useSessionStore()
const model = ref(session.selectedModel || 'graphsage')
const topK = ref(5)

const isTemporal = computed(() => model.value === 'tgat' || model.value === 'tgn')

function barWidth(feat: ExplainFeature, result: ExplainResult): string {
  const maxImportance = Math.max(...result.top_features.map(f => Math.abs(f.importance)))
  if (maxImportance === 0) return '0%'
  return `${(Math.abs(feat.importance) / maxImportance) * 100}%`
}

async function runTopK() {
  session.selectedModel = model.value
  await session.explainTopAlerts(model.value, topK.value)
}
</script>

<style scoped>
.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 12px;
}
h2 { font-size: 18px; }
.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.controls select {
  padding: 6px 10px;
  background: #1e293b;
  border: 1px solid #475569;
  border-radius: 6px;
  color: #f1f5f9;
  font-size: 13px;
}
.top-k-group {
  display: flex;
  align-items: center;
  gap: 6px;
}
.top-k-group label { font-size: 13px; color: #94a3b8; }
.top-k-group input {
  width: 56px;
  padding: 6px 8px;
  background: #1e293b;
  border: 1px solid #475569;
  border-radius: 6px;
  color: #f1f5f9;
  font-size: 13px;
  text-align: center;
}
.btn-explain {
  padding: 6px 16px;
  background: #7c3aed;
  border: none;
  border-radius: 6px;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.btn-explain:hover:not(:disabled) { background: #6d28d9; }
.btn-explain:disabled { opacity: 0.5; cursor: not-allowed; }

.loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 0;
  gap: 16px;
  color: #94a3b8;
}
.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #334155;
  border-top-color: #7c3aed;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #94a3b8;
}
.empty-state .hint { font-size: 13px; color: #64748b; margin-top: 8px; }

.results {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.explain-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  flex-wrap: wrap;
  font-size: 13px;
}
.rank {
  background: #7c3aed;
  color: #fff;
  padding: 2px 8px;
  border-radius: 9999px;
  font-weight: 700;
  font-size: 12px;
}
.edge-id { font-family: monospace; color: #94a3b8; }
.nodes { font-family: monospace; color: #60a5fa; }
.timestamp { color: #64748b; font-family: monospace; }
.class-badge {
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 600;
}
.class-badge.attack { background: #450a0a; color: #fca5a5; }
.class-badge.benign { background: #052e16; color: #86efac; }
.confidence { font-weight: 600; color: #f1f5f9; }
.method-tag {
  background: #1e1b4b;
  color: #a5b4fc;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-family: monospace;
}

.feature-bars { margin-top: 4px; }
.bar-label-row { margin-bottom: 10px; }
.bar-title { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

.bar-row {
  display: grid;
  grid-template-columns: 140px 1fr 60px;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.feat-name {
  font-family: monospace;
  font-size: 12px;
  color: #94a3b8;
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bar-track {
  height: 18px;
  background: #0f172a;
  border-radius: 3px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.4s ease-out;
}
.bar-fill.positive { background: #7c3aed; }
.bar-fill.negative { background: #dc2626; }
.feat-value {
  font-family: monospace;
  font-size: 12px;
  color: #64748b;
  text-align: right;
}

.meta-row {
  font-size: 12px;
  color: #64748b;
  margin-top: 8px;
}
.meta-row strong { color: #94a3b8; }
</style>
