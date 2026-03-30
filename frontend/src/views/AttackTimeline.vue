<template>
  <div class="view-header">
    <h2>Attack Timeline</h2>
    <span class="meta">{{ session.timeline?.x.length ?? 0 }} × 60-second windows</span>
  </div>
  <div v-if="!session.timeline" class="empty-state">
    Run analysis first to see the attack timeline.
  </div>
  <div v-else ref="plotContainer" class="plot-container" />
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const plotContainer = ref<HTMLElement | null>(null)

async function renderPlot() {
  if (!plotContainer.value || !session.timeline) return
  // Dynamic import keeps plotly out of the initial bundle
  const Plotly = (await import('plotly.js-basic-dist-min')).default

  const data = session.timeline.traces.map((trace) => ({
    x: session.timeline!.x,
    y: trace.y,
    name: trace.name,
    type: 'bar' as const,
    marker: { color: trace.colour },
  }))

  Plotly.react(plotContainer.value, data, {
    barmode: 'stack',
    paper_bgcolor: '#0f172a',
    plot_bgcolor: '#0f172a',
    font: { color: '#f1f5f9', size: 12 },
    xaxis: { title: 'Time Window (60 s)', gridcolor: '#334155', zerolinecolor: '#334155' },
    yaxis: { title: 'Flow Count', gridcolor: '#334155', zerolinecolor: '#334155' },
    legend: { bgcolor: '#1e293b', bordercolor: '#334155', borderwidth: 1 },
    margin: { t: 20, r: 20, b: 60, l: 60 },
  })
}

watch(() => session.timeline, renderPlot)
onMounted(renderPlot)

onUnmounted(async () => {
  if (!plotContainer.value) return
  const Plotly = (await import('plotly.js-basic-dist-min')).default
  Plotly.purge(plotContainer.value)
})
</script>

<style scoped>
.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
h2 { font-size: 18px; }
.meta { font-size: 13px; color: #64748b; }
.plot-container { width: 100%; height: calc(100vh - 180px); }
.empty-state { color: #64748b; text-align: center; margin-top: 80px; font-size: 15px; }
</style>
