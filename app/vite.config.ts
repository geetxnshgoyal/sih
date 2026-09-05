import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { defineConfig } from 'vite'

const useHttps = process.env.VITE_DEV_HTTPS === '1'

// https://vite.dev/config/
export default defineConfig({
  // Localhost gets the camera on plain HTTP because browsers treat it as a
  // secure context. Use VITE_DEV_HTTPS=1 for LAN/phone demos, where HTTPS is
  // required and the browser will show a one-time self-signed certificate
  // warning.
  //
  // Example: VITE_DEV_HTTPS=1 npm run dev
  plugins: [tailwindcss(), react(), ...(useHttps ? [basicSsl()] : [])],
  server: {
    host: useHttps ? true : '127.0.0.1',
    port: 5174,
  },
})

