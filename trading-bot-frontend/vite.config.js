import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // This is important for Docker and network access
    port: 5173,
    strictPort: true,
    watch: {
      usePolling: true
    },
    allowedHosts: [
      '5173-ixyq9i42cuaujsqf3wc01-4fcd64da.manusvm.computer' // Add the exposed host here
    ]
  }
})


