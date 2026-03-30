<template>
  <div class="view-header">
    <h2>Traffic Graph</h2>
    <div class="header-actions">
      <button class="btn-sm" @click="exportPng">Export PNG</button>
      <span class="meta">{{ session.graphNodes.length }} nodes · {{ session.graphEdges.length }} edges</span>
    </div>
  </div>
  <div ref="cyContainer" class="cy-container" />
  <div v-if="selectedNode" class="sidebar">
    <h3>{{ selectedNode.data.ip }}</h3>
    <p>Risk score: <strong>{{ (selectedNode.data.riskScore * 100).toFixed(1) }}%</strong></p>
    <p>Incident alerts: {{ incidentAlerts.length }}</p>
    <ul class="alert-mini-list">
      <li v-for="a in incidentAlerts.slice(0, 5)" :key="a.flow_id">
        <span class="badge-attack">{{ a.attack_type }}</span> {{ (a.confidence * 100).toFixed(0) }}%
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const cyContainer = ref<HTMLElement | null>(null)
const selectedNode = ref<{ data: { id: string; ip: string; riskScore: number } } | null>(null)

let cy: ReturnType<typeof import('cytoscape')['default']> | null = null

const incidentAlerts = computed(() =>
  session.alerts.filter(
    a => a.src === selectedNode.value?.data.id || a.dst === selectedNode.value?.data.id
  )
)

async function initCytoscape() {
  if (!cyContainer.value || !session.graphNodes.length) return
  const Cytoscape = (await import('cytoscape')).default

  cy = Cytoscape({
    container: cyContainer.value,
    elements: {
      nodes: session.graphNodes.map(n => ({
        data: n.data,
        position: { x: n.data.x, y: n.data.y },
      })),
      edges: session.graphEdges.map(e => ({ data: e.data })),
    },
    layout: { name: 'preset' },
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'data(colour)',
          'width': 20, 'height': 20,
          'label': 'data(ip)',
          'font-size': 8, 'color': '#f1f5f9',
          'text-valign': 'bottom', 'text-margin-y': 4,
        },
      },
      {
        selector: 'edge',
        style: {
          'line-color': 'data(colour)',
          'width': 'data(width)',
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': 'data(colour)',
          'arrow-scale': 0.8,
        },
      },
    ],
  })

  cy.on('tap', 'node', (e) => {
    selectedNode.value = { data: e.target.data() }
  })
  cy.on('tap', (e) => {
    if (e.target === cy) selectedNode.value = null
  })
}

function exportPng() {
  if (!cy) return
  const b64 = cy.png({ output: 'base64', scale: 2, full: true })
  // Store in session for report generation
  sessionStorage.setItem('graph_png_b64', b64)
  // Also trigger download
  const link = document.createElement('a')
  link.href = `data:image/png;base64,${b64}`
  link.download = 'traffic_graph.png'
  link.click()
}

watch(() => session.graphNodes, initCytoscape, { immediate: true })
onMounted(initCytoscape)
onUnmounted(() => { cy?.destroy(); cy = null })
</script>

<style scoped>
.view-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
h2 { font-size: 18px; }
.header-actions { display: flex; align-items: center; gap: 12px; }
.btn-sm { padding: 6px 12px; background: #334155; border: none; border-radius: 6px; color: #f1f5f9; font-size: 13px; cursor: pointer; }
.btn-sm:hover { background: #475569; }
.meta { font-size: 13px; color: #64748b; }
.cy-container { width: 100%; height: calc(100vh - 180px); background: #0f172a; border: 1px solid #334155; border-radius: 8px; }
.sidebar { position: fixed; right: 24px; top: 80px; width: 240px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; }
.sidebar h3 { font-size: 14px; margin-bottom: 8px; word-break: break-all; }
.sidebar p { font-size: 13px; color: #94a3b8; margin: 4px 0; }
.alert-mini-list { list-style: none; margin-top: 10px; }
.alert-mini-list li { font-size: 12px; padding: 4px 0; display: flex; align-items: center; gap: 6px; }
.badge-attack { background: #fee2e2; color: #b91c1c; padding: 1px 6px; border-radius: 9999px; font-size: 11px; }
</style>
