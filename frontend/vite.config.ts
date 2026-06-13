import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 开发模式:/api 转发给 Rust 网关(生产则由网关直接托管 dist)
    proxy: {
      "/api": "http://localhost:3001",
    },
  },
})
