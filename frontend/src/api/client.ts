import axios from 'axios'

const baseURL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: `${baseURL}/api`,
  timeout: 60_000,
})
