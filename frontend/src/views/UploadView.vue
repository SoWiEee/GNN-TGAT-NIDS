<template>
  <div class="upload-view">
    <div class="upload-card">
      <h1>GNN-NIDS Analyzer</h1>
      <p class="subtitle">Upload a NetFlow CSV to detect intrusions with Graph Neural Networks</p>

      <div
        class="dropzone"
        :class="{ 'dropzone--active': isDragging, 'dropzone--error': session.status === 'error' }"
        @dragover.prevent="isDragging = true"
        @dragleave="isDragging = false"
        @drop.prevent="onDrop"
        @click="fileInput?.click()"
      >
        <input ref="fileInput" type="file" accept=".csv" style="display:none" @change="onFileChange" />
        <div v-if="!selectedFile">
          <p class="drop-icon">📄</p>
          <p>Drop a NetFlow CSV here or <strong>click to browse</strong></p>
          <p class="hint">Max 50 MB · NF-UNSW-NB15-v2 format</p>
        </div>
        <div v-else>
          <p class="drop-icon">✅</p>
          <p><strong>{{ selectedFile.name }}</strong></p>
          <p class="hint">{{ (selectedFile.size / 1024 / 1024).toFixed(2) }} MB</p>
        </div>
      </div>

      <div class="model-select">
        <label>Model</label>
        <select v-model="selectedModel">
          <option value="gat">GAT (Graph Attention Network)</option>
          <option value="graphsage">GraphSAGE</option>
        </select>
      </div>

      <button
        class="btn-primary"
        :disabled="!selectedFile || session.status === 'uploading' || session.status === 'analyzing'"
        @click="submit"
      >
        <span v-if="session.status === 'uploading'">Uploading…</span>
        <span v-else-if="session.status === 'analyzing'">
          Analyzing… {{ session.progressPct.toFixed(0) }}%
        </span>
        <span v-else>Analyze</span>
      </button>

      <div v-if="session.status === 'analyzing'" class="progress-bar">
        <div class="progress-fill" :style="{ width: session.progressPct + '%' }" />
      </div>

      <p v-if="session.status === 'error'" class="error-msg">{{ session.errorMessage }}</p>
    </div>

    <div class="demo-hint">
      <strong>Demo:</strong> Use <code>data/demo/demo_flows.csv</code> to try without downloading the full dataset.
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const router = useRouter()

const selectedFile = ref<File | null>(null)
const selectedModel = ref('gat')
const isDragging = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

function onDrop(e: DragEvent) {
  isDragging.value = false
  const file = e.dataTransfer?.files[0]
  if (file?.name.endsWith('.csv')) selectedFile.value = file
}

function onFileChange(e: Event) {
  const input = e.target as HTMLInputElement
  selectedFile.value = input.files?.[0] ?? null
}

async function submit() {
  if (!selectedFile.value) return
  await session.uploadAndAnalyze(selectedFile.value, selectedModel.value)
}

watch(() => session.status, (s) => {
  if (s === 'ready') router.push('/graph')
})
</script>

<style scoped>
.upload-view { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 70vh; gap: 20px; }
.upload-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 40px; width: 100%; max-width: 560px; }
h1 { font-size: 24px; margin-bottom: 8px; }
.subtitle { color: #94a3b8; margin-bottom: 28px; }
.dropzone { border: 2px dashed #475569; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: all .2s; }
.dropzone:hover, .dropzone--active { border-color: #60a5fa; background: #1e3a5f22; }
.dropzone--error { border-color: #ef4444; }
.drop-icon { font-size: 32px; margin-bottom: 10px; }
.hint { font-size: 12px; color: #64748b; margin-top: 6px; }
.model-select { margin: 20px 0; }
.model-select label { display: block; font-size: 13px; color: #94a3b8; margin-bottom: 6px; }
.model-select select { width: 100%; padding: 8px 12px; background: #0f172a; border: 1px solid #475569; border-radius: 6px; color: #f1f5f9; font-size: 14px; }
.btn-primary { width: 100%; padding: 12px; background: #3b82f6; border: none; border-radius: 8px; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; transition: background .2s; }
.btn-primary:hover:not(:disabled) { background: #2563eb; }
.btn-primary:disabled { opacity: .5; cursor: not-allowed; }
.progress-bar { height: 6px; background: #334155; border-radius: 3px; margin-top: 12px; overflow: hidden; }
.progress-fill { height: 100%; background: #3b82f6; transition: width .3s; }
.error-msg { color: #f87171; font-size: 13px; margin-top: 12px; }
.demo-hint { font-size: 13px; color: #64748b; }
.demo-hint code { background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
</style>
