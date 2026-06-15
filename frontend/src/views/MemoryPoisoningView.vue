<template>
  <div class="memory-view">
    <div class="header-row">
      <div>
        <h2 class="page-title">Memory Poisoning</h2>
        <p class="subtitle">Temporal attack replay against TGAT/TGN node memory.</p>
      </div>
      <button class="btn-primary" :disabled="loading" @click="run">
        {{ loading ? 'Running...' : 'Run' }}
      </button>
    </div>

    <section class="controls">
      <label>
        Model
        <select v-model="form.model">
          <option value="tgn">TGN</option>
          <option value="tgat">TGAT</option>
        </select>
      </label>
      <label>
        Strategy
        <select v-model="form.poison_strategy">
          <option value="benign_mean">Benign mean</option>
          <option value="random_benign">Random benign</option>
        </select>
      </label>
      <label>
        Poison / node
        <input v-model.number="form.n_poison" type="number" min="0" max="20" />
      </label>
      <label>
        Batches
        <input v-model.number="form.max_batches" type="number" min="1" max="200" />
      </label>
      <label>
        Batch size
        <input v-model.number="form.batch_size" type="number" min="10" max="2000" />
      </label>
    </section>

    <div v-if="error" class="error-msg">{{ error }}</div>

    <section v-if="result" class="summary-grid">
      <div class="summary-item">
        <span>ASR</span>
        <strong>{{ result.asr.toFixed(3) }}</strong>
      </div>
      <div class="summary-item">
        <span>Attack Edges</span>
        <strong>{{ result.total_attack_edges }}</strong>
      </div>
      <div class="summary-item">
        <span>Evaded</span>
        <strong>{{ result.total_evaded }}</strong>
      </div>
      <div class="summary-item">
        <span>Poison Events</span>
        <strong>{{ result.total_poison_events }}</strong>
      </div>
    </section>

    <section v-if="result" class="batch-list">
      <div v-for="row in result.rows" :key="row.batch" class="batch-row">
        <div class="batch-meta">
          <strong>Batch {{ row.batch }}</strong>
          <span>{{ row.attack_edges }} attacks · {{ row.evaded }} evaded · {{ row.poison_events }} poison</span>
        </div>
        <div class="bar">
          <div class="bar-fill" :style="{ width: row.asr * 100 + '%' }" />
        </div>
        <span class="asr">{{ row.asr.toFixed(3) }}</span>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { api, type MemoryPoisoningResult } from '@/api'

const form = reactive({
  model: 'tgn',
  poison_strategy: 'benign_mean',
  n_poison: 3,
  max_batches: 20,
  batch_size: 200,
})
const loading = ref(false)
const error = ref('')
const result = ref<MemoryPoisoningResult | null>(null)

async function run() {
  loading.value = true
  error.value = ''
  try {
    const res = await api.runMemoryPoisoning(form)
    result.value = res.data
  } catch (err: any) {
    error.value = err?.response?.data?.detail ?? 'Memory poisoning experiment failed.'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.memory-view { display: flex; flex-direction: column; gap: 20px; }
.header-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.page-title { font-size: 18px; margin-bottom: 8px; }
.subtitle { font-size: 13px; color: #94a3b8; }
.controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 14px;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 18px;
}
label { display: flex; flex-direction: column; gap: 6px; color: #94a3b8; font-size: 12px; }
select, input {
  width: 100%;
  padding: 8px 10px;
  background: #0f172a;
  border: 1px solid #475569;
  border-radius: 6px;
  color: #f1f5f9;
}
.btn-primary {
  min-width: 100px;
  padding: 10px 14px;
  background: #3b82f6;
  border: 0;
  border-radius: 8px;
  color: #fff;
  font-weight: 600;
  cursor: pointer;
}
.btn-primary:disabled { opacity: .55; cursor: not-allowed; }
.error-msg { color: #f87171; font-size: 13px; }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.summary-item {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 16px;
}
.summary-item span { display: block; color: #94a3b8; font-size: 12px; margin-bottom: 6px; }
.summary-item strong { font-size: 24px; }
.batch-list { display: flex; flex-direction: column; gap: 10px; }
.batch-row {
  display: grid;
  grid-template-columns: minmax(180px, 280px) 1fr 54px;
  gap: 14px;
  align-items: center;
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 12px;
}
.batch-meta { display: flex; flex-direction: column; gap: 4px; }
.batch-meta strong { font-size: 13px; }
.batch-meta span { color: #94a3b8; font-size: 12px; }
.bar { height: 10px; background: #334155; border-radius: 5px; overflow: hidden; }
.bar-fill { height: 100%; background: #f97316; transition: width .35s ease; }
.asr { text-align: right; font-weight: 700; }
@media (max-width: 720px) {
  .header-row { flex-direction: column; }
  .batch-row { grid-template-columns: 1fr; }
  .asr { text-align: left; }
}
</style>
