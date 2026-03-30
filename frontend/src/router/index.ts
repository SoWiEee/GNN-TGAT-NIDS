import { createRouter, createWebHistory } from 'vue-router'
import UploadView from '@/views/UploadView.vue'
import TrafficGraph from '@/views/TrafficGraph.vue'
import AlertList from '@/views/AlertList.vue'
import AttackTimeline from '@/views/AttackTimeline.vue'
import ReliabilityPanel from '@/views/ReliabilityPanel.vue'
import AdversarialReport from '@/views/AdversarialReport.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: UploadView },
    { path: '/graph', component: TrafficGraph },
    { path: '/alerts', component: AlertList },
    { path: '/timeline', component: AttackTimeline },
    { path: '/reliability', component: ReliabilityPanel },
    { path: '/adversarial', component: AdversarialReport },
  ],
})

export default router
