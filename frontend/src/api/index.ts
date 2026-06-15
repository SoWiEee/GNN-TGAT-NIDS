import { apiClient } from './client'
import type { AxiosResponse } from 'axios'

export interface UploadResponse {
  session_id: string
  n_flows: number
  filename: string
}

export interface StatusResponse {
  session_id: string
  status: 'idle' | 'analyzing' | 'ready' | 'error'
  progress_pct: number
  message: string
}

export interface Alert {
  flow_id: string
  src: string
  dst: string
  attack_type: string
  confidence: number
  top_features: Array<{ name: string; value: number }>
  window: number
}

export interface CyNode {
  data: { id: string; ip: string; riskScore: number; colour: string; x: number; y: number }
}

export interface CyEdge {
  data: {
    id: string; source: string; target: string
    prediction: string; confidence: number; flowId: string; colour: string; width: number
  }
}

export interface GraphResponse {
  nodes: CyNode[]
  edges: CyEdge[]
}

export interface TimelineResponse {
  x: number[]
  traces: Array<{ name: string; y: number[]; colour: string }>
}

export interface AdversarialResult {
  flow_id: string
  original: { prediction: string; confidence: number; features: Record<string, number> }
  adversarial: {
    prediction: string
    confidence: number | null
    features: Record<string, number>
    csr: number
    changed_features: Array<{ name: string; original: number; adversarial: number; delta_pct: number; constraint_ok: boolean }>
  }
}

export interface PerClassMetric {
  class_id: number
  name: string
  precision: number
  recall: number
  f1: number
  support: number
}

export interface ReliabilityMetrics {
  [model: string]: {
    clean_f1: number | null
    clean_precision?: number | null
    clean_recall?: number | null
    clean_roc_auc?: number | null
    clean_macro_f1?: number | null
    per_class?: PerClassMetric[] | null
    confusion_matrix?: number[][] | null
    class_names?: string[] | null
    dr_under_cpgd_eps01: number | null
    dr_under_cpgd_eps01_sampled?: boolean | null
    cpgd_epsilon?: number | null
    cpgd_steps?: number | null
    cpgd_sample_windows?: number | null
    cpgd_attack_edges?: number | null
    cpgd_scope?: string | null
    cpgd_constraint?: string | null
    delta_f1_after_adv_training: number | null
  }
}

export interface MemoryPoisoningRow {
  batch: number
  events: number
  attack_edges: number
  evaded: number
  poison_events: number
  asr: number
}

export interface MemoryPoisoningResult {
  model: string
  n_poison: number
  poison_strategy: string
  batches: number
  total_attack_edges: number
  total_evaded: number
  total_poison_events: number
  asr: number
  rows: MemoryPoisoningRow[]
}

export const api = {
  upload(file: File): Promise<AxiosResponse<UploadResponse>> {
    const form = new FormData()
    form.append('file', file)
    return apiClient.post('/upload', form)
  },

  analyze(sessionId: string, model: string): Promise<AxiosResponse<{ session_id: string; status: string }>> {
    return apiClient.post(`/analyze/${sessionId}`, { model })
  },

  getStatus(sessionId: string): Promise<AxiosResponse<StatusResponse>> {
    return apiClient.get(`/status/${sessionId}`)
  },

  getGraph(sessionId: string, maxEdges = 2000): Promise<AxiosResponse<GraphResponse>> {
    return apiClient.get(`/graph/${sessionId}`, { params: { max_edges: maxEdges } })
  },

  getAlerts(
    sessionId: string,
    params: { sort?: string; page?: number; limit?: number; attack_type?: string } = {}
  ): Promise<AxiosResponse<{ alerts: Alert[]; total: number }>> {
    return apiClient.get(`/alerts/${sessionId}`, { params })
  },

  getTimeline(sessionId: string): Promise<AxiosResponse<TimelineResponse>> {
    return apiClient.get(`/timeline/${sessionId}`)
  },

  generateAdversarial(
    sessionId: string, flowId: string, epsilon: number, steps: number
  ): Promise<AxiosResponse<AdversarialResult>> {
    return apiClient.post('/adversarial', { session_id: sessionId, flow_id: flowId, epsilon, steps })
  },

  generateReport(sessionId: string, graphPngB64 = ''): Promise<AxiosResponse<{ report_url: string }>> {
    return apiClient.post(`/report/${sessionId}`, { session_id: sessionId, graph_png_b64: graphPngB64 })
  },

  getMetrics(): Promise<AxiosResponse<ReliabilityMetrics>> {
    return apiClient.get('/metrics')
  },

  runMemoryPoisoning(params: {
    model: string
    n_poison: number
    max_batches: number
    batch_size: number
    poison_strategy: string
  }): Promise<AxiosResponse<MemoryPoisoningResult>> {
    return apiClient.post('/memory-poisoning', params)
  },
}
