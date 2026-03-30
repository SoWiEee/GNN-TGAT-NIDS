import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api } from '@/api'
import type { Alert, CyNode, CyEdge, TimelineResponse, ReliabilityMetrics, AdversarialResult } from '@/api'

export const useSessionStore = defineStore('session', () => {
  const sessionId = ref<string | null>(null)
  const status = ref<'idle' | 'uploading' | 'analyzing' | 'ready' | 'error'>('idle')
  const progressPct = ref(0)
  const errorMessage = ref('')

  const graphNodes = ref<CyNode[]>([])
  const graphEdges = ref<CyEdge[]>([])
  const alerts = ref<Alert[]>([])
  const totalAlerts = ref(0)
  const timeline = ref<TimelineResponse | null>(null)
  const reliability = ref<ReliabilityMetrics | null>(null)
  const adversarialResult = ref<AdversarialResult | null>(null)
  const selectedFlowId = ref<string | null>(null)

  const isReady = computed(() => status.value === 'ready')

  // --- Polling ---
  let pollingTimer: ReturnType<typeof setInterval> | null = null

  function _stopPolling() {
    if (pollingTimer !== null) {
      clearInterval(pollingTimer)
      pollingTimer = null
    }
  }

  function _startPolling(sid: string) {
    _stopPolling()
    pollingTimer = setInterval(async () => {
      try {
        const { data } = await api.getStatus(sid)
        status.value = data.status
        progressPct.value = data.progress_pct
        if (data.status === 'ready') {
          _stopPolling()
          await _loadResults(sid)
        } else if (data.status === 'error') {
          errorMessage.value = data.message
          _stopPolling()
        }
      } catch {
        _stopPolling()
        status.value = 'error'
        errorMessage.value = 'Lost connection to server'
      }
    }, 2000)
  }

  async function _loadResults(sid: string) {
    const [graphRes, alertsRes, timelineRes] = await Promise.all([
      api.getGraph(sid),
      api.getAlerts(sid),
      api.getTimeline(sid),
    ])
    graphNodes.value = graphRes.data.nodes
    graphEdges.value = graphRes.data.edges
    alerts.value = alertsRes.data.alerts
    totalAlerts.value = alertsRes.data.total
    timeline.value = timelineRes.data
  }

  // --- Actions ---
  async function uploadAndAnalyze(file: File, model: string) {
    _stopPolling()
    sessionId.value = null
    status.value = 'uploading'
    errorMessage.value = ''
    progressPct.value = 0

    try {
      const { data: uploadData } = await api.upload(file)
      sessionId.value = uploadData.session_id
      status.value = 'analyzing'
      await api.analyze(uploadData.session_id, model)
      _startPolling(uploadData.session_id)
    } catch (err: unknown) {
      status.value = 'error'
      errorMessage.value = err instanceof Error ? err.message : 'Upload failed'
    }
  }

  async function loadMoreAlerts(page: number, attackType = '') {
    if (!sessionId.value) return
    const { data } = await api.getAlerts(sessionId.value, { page, limit: 50, attack_type: attackType })
    alerts.value = data.alerts
    totalAlerts.value = data.total
  }

  async function generateAdversarial(flowId: string, epsilon: number, steps: number) {
    if (!sessionId.value) return
    selectedFlowId.value = flowId
    adversarialResult.value = null
    const { data } = await api.generateAdversarial(sessionId.value, flowId, epsilon, steps)
    adversarialResult.value = data
  }

  async function loadReliability() {
    const { data } = await api.getMetrics()
    reliability.value = data
  }

  function reset() {
    _stopPolling()
    sessionId.value = null
    status.value = 'idle'
    progressPct.value = 0
    errorMessage.value = ''
    graphNodes.value = []
    graphEdges.value = []
    alerts.value = []
    totalAlerts.value = 0
    timeline.value = null
    adversarialResult.value = null
    selectedFlowId.value = null
  }

  return {
    sessionId, status, progressPct, errorMessage,
    graphNodes, graphEdges, alerts, totalAlerts, timeline, reliability,
    adversarialResult, selectedFlowId, isReady,
    uploadAndAnalyze, loadMoreAlerts, generateAdversarial, loadReliability, reset,
  }
})
