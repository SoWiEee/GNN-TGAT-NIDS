<template>
  <div class="view-header">
    <h2>Alert List</h2>
    <div class="filters">
      <select v-model="attackFilter" @change="loadAlerts(1)">
        <option value="">All attack types</option>
        <option v-for="t in attackTypes" :key="t" :value="t">{{ t }}</option>
      </select>
      <span class="meta">{{ session.totalAlerts }} alerts</span>
    </div>
  </div>

  <table class="alert-table">
    <thead>
      <tr>
        <th>Flow ID</th>
        <th>Src → Dst</th>
        <th>Attack Type</th>
        <th>Confidence</th>
        <th>Top Features</th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="alert in session.alerts" :key="alert.flow_id">
        <td class="mono">{{ alert.flow_id }}</td>
        <td class="mono small">{{ alert.src }} → {{ alert.dst }}</td>
        <td><span class="badge-attack">{{ alert.attack_type }}</span></td>
        <td>{{ (alert.confidence * 100).toFixed(1) }}%</td>
        <td class="features">
          <span v-for="f in alert.top_features.slice(0, 3)" :key="f.name" class="feat-tag">
            {{ f.name }}
          </span>
        </td>
        <td>
          <button class="btn-adv" @click="goAdversarial(alert.flow_id)">Adversarial →</button>
        </td>
      </tr>
      <tr v-if="!session.alerts.length">
        <td colspan="6" class="empty-row">No alerts detected in this capture.</td>
      </tr>
    </tbody>
  </table>

  <div class="pagination">
    <button :disabled="page === 1" @click="loadAlerts(page - 1)">← Prev</button>
    <span>Page {{ page }}</span>
    <button :disabled="session.alerts.length < pageSize" @click="loadAlerts(page + 1)">Next →</button>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const router = useRouter()
const attackFilter = ref('')
const page = ref(1)
const pageSize = 50

const attackTypes = computed(() => [...new Set(session.alerts.map((a) => a.attack_type))])

async function loadAlerts(p: number) {
  page.value = p
  await session.loadMoreAlerts(p, attackFilter.value)
}

function goAdversarial(flowId: string) {
  session.selectedFlowId = flowId
  router.push('/adversarial')
}

onMounted(() => loadAlerts(1))
</script>

<style scoped>
.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
h2 { font-size: 18px; }
.filters { display: flex; align-items: center; gap: 12px; }
.filters select {
  padding: 6px 10px;
  background: #1e293b;
  border: 1px solid #475569;
  border-radius: 6px;
  color: #f1f5f9;
  font-size: 13px;
}
.meta { font-size: 13px; color: #64748b; }
.alert-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.alert-table th {
  background: #1e293b;
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #334155;
  font-size: 12px;
  color: #94a3b8;
  text-transform: uppercase;
}
.alert-table td { padding: 10px 12px; border-bottom: 1px solid #1e293b; }
.alert-table tr:hover td { background: #1e293b44; }
.mono { font-family: monospace; }
.small { font-size: 11px; }
.badge-attack {
  background: #450a0a;
  color: #fca5a5;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 12px;
}
.feat-tag {
  background: #1e293b;
  border: 1px solid #334155;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 11px;
  margin-right: 4px;
  font-family: monospace;
}
.btn-adv {
  padding: 4px 10px;
  background: #1d4ed8;
  border: none;
  border-radius: 6px;
  color: #fff;
  font-size: 12px;
  cursor: pointer;
}
.btn-adv:hover { background: #2563eb; }
.empty-row { color: #64748b; text-align: center; padding: 32px; }
.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-top: 20px;
  font-size: 14px;
}
.pagination button {
  padding: 6px 14px;
  background: #334155;
  border: none;
  border-radius: 6px;
  color: #f1f5f9;
  cursor: pointer;
}
.pagination button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
