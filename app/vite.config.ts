import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Served from a subpath on GitHub Pages (/<repo>/), from the root in dev and
  // on a root-domain host. Vite bakes this into import.meta.env.BASE_URL, which
  // lib/assetUrl.ts uses to resolve the model files — so the same source builds
  // correctly for either without a code change.
  base: process.env.DEPLOY_BASE ?? "/",

  // HTTPS is not optional here, even on a LAN.
  //
  // getUserMedia only runs in a "secure context". Browsers exempt localhost, so
  // http://localhost:5174 gets the camera on this machine — but a phone hitting
  // http://<lan-ip>:5174 is NOT exempt, and the camera is refused. The page
  // loads fine and the button simply does nothing, which is a miserable thing
  // to debug on stage.
  //
  // basicSsl issues a self-signed cert so the LAN origin counts as secure. The
  // phone warns once ("connection is not private") because the cert is not from
  // a public CA — accepting it is expected, and nothing leaves the local network.
  plugins: [react(), basicSsl()],
  server: {
    // Bind all interfaces, not just loopback, so the phone can reach it.
    host: true,
    port: 5174,
  },
})
