import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api } from '@/api'
import type { Alert, CyNode, CyEdge, TimelineResponse, ReliabilityMetrics, AdversarialResult, ExplainResult } from '@/api'

const MAX_UPLOAD_BYTES = Number(import.meta.env.VITE_MAX_UPLOAD_BYTES ?? 50 * 1024 * 1024)
const POLL_INTERVAL_MS = Number(import.meta.env.VITE_POLL_INTERVAL_MS ?? 2000)

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
  const explainResults = ref<ExplainResult[]>([])
  const explainLoading = ref(false)
  const selectedModel = ref('graphsage')

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
    }, POLL_INTERVAL_MS)
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

    if (file.size > MAX_UPLOAD_BYTES) {
      status.value = 'error'
      errorMessage.value = `File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB. Max is 50 MB.`
      return
    }

    try {
      const { data: uploadData } = await api.upload(file)
      sessionId.value = uploadData.session_id
      status.value = 'analyzing'
      await api.analyze(uploadData.session_id, model)
      _startPolling(uploadData.session_id)
    } catch (err: unknown) {
      status.value = 'error'
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } }
        errorMessage.value = axiosErr.response?.data?.detail ?? 'Upload failed'
      } else {
        errorMessage.value = err instanceof Error ? err.message : 'Upload failed'
      }
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

  async function explainFlow(edgeIdx: number, model: string, epochs = 200) {
    if (!sessionId.value) return
    explainLoading.value = true
    try {
      const { data } = await api.explainFlow(sessionId.value, model, edgeIdx, epochs)
      explainResults.value = [data]
    } finally {
      explainLoading.value = false
    }
  }

  async function explainTopAlerts(model: string, topK = 5, epochs = 200) {
    if (!sessionId.value) return
    explainLoading.value = true
    try {
      const { data } = await api.explainTopAlerts(sessionId.value, model, topK, epochs)
      explainResults.value = data
    } finally {
      explainLoading.value = false
    }
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
    explainResults.value = []
    explainLoading.value = false
  }

  return {
    sessionId, status, progressPct, errorMessage,
    graphNodes, graphEdges, alerts, totalAlerts, timeline, reliability,
    adversarialResult, selectedFlowId, isReady,
    explainResults, explainLoading, selectedModel,
    uploadAndAnalyze, loadMoreAlerts, generateAdversarial, loadReliability,
    explainFlow, explainTopAlerts, reset,
  }
})
